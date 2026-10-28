const { ConvolveForward, fusedFinalDeltaProjector_ForConv, ConvolveBackward } = require("../../core/bindings");
const {  calculateTensorShape, createTensorBuffer } = require("../../utils/utils");
const errFlags = require('../../err_flags');



/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>}}
 */
const initParams = (size, shape, layer_data) => {
    const filters = layer_data.filters;
    const [kH, kW] = layer_data.kernel_size;
    const stride = layer_data.strides || 1;
    const padding = layer_data.padding || "same";
    const useBias = layer_data.useBias;

    const inputH = shape[0];
    const inputW = shape[1];
    const inputDepth = shape[2];

    const inputShape = [inputH, inputW, inputDepth];

    const TotalSize = filters * kH * kW * inputDepth;

    const fanIn = kH * kW * inputDepth;
    const fanOut = kH * kW * filters;

    let kernels = createTensorBuffer([TotalSize], {prefilledWith:'xavier', min: fanIn, max: fanOut}).data;
    let kernelGrads = createTensorBuffer([TotalSize], {prefilledWith:'zeroes'}).data;
    let biases = createTensorBuffer([filters], {prefilledWith:'zeroes'}).data;
    let biasGrads = createTensorBuffer([filters], {prefilledWith:'zeroes'}).data;

    if (useBias) {
        biases = createTensorBuffer([filters], {prefilledWith:'xavier', min: fanIn, max: fanOut}).data;
    }
    

    // Calculate output shape
    const { OutputHeight, OutputWidth, CalculatedTensorShape } = calculateTensorShape(inputH, inputW, kH, kW, filters, stride, padding);
    // store output shape too
    const outputShape = [OutputHeight, OutputWidth, filters];

    const weightShape = [filters, kH, kW, inputDepth];
                    
    return {
        updatedSize: CalculatedTensorShape,
        updatedShape: outputShape,
        weights: kernels,
        biases: biases,
        weightGrads: kernelGrads,
        biasGrads: biasGrads,
        inputShape: inputShape,
        outputShape: outputShape,
        paramShape: weightShape,
    }
    
}

const determineInferenceType = () => {
    throw new Error('Convolutional layer cannot be an output layer for now');
}

const feedforward = (data) => {
    const layerData = data.layerData;
    const kernelShape = layerData.weightShape;
    const inputShape= layerData.inputShape;
    const outputShape = layerData.outputShape;
    const input = data.input;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    if (input.some(Number.isNaN)) {
        console.error("Input has NaNs detected during feedforward in convolutional layer");
        throw new Error(errFlags.NAN_ERR);
    }

    const totalSize = layerData.inputShape.reduce((acc, val) => acc * val, 1);

    if (input.length != totalSize) {
        console.error(`Input tensor doesn't match with the expected input tensor shape: Expected shape/size: ${[input_H, input_W, input_D]} or ${totalSize}. The size of the input entered is ${input.length}`);
        throw new Error(errFlags.SHAPE_ERR);
    }

    const activationFuncString = layerData.activation_function.name;
    const {z_values, activated_output} = ConvolveForward(input, inputShape, outputShape, kernelShape, activationFuncString, pointer, modelID, layerID);

    if (activated_output.some(v => Number.isNaN(v))) {
        console.error("NaN detected after activation function during feedforward in convolutional layer");
        throw new Error(errFlags.NAN_ERR);
    }

    layerData.cache = {
        layer_output: activated_output,
    }

    return {
        outputs: activated_output,
        z_values: z_values,
        incrementor_value: 1
    };
}

const getOutputLayerDelta = () => {
    throw new Error('Convolutional layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
}

const projectDeltaBackward = (data) => {

    const layerData = data.layerData;
    const delta = data.delta;
    const targetShape = layerData.inputShape;
    const deltaShape = layerData.outputShape;
    const kernelShape = layerData.weightShape;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    if (delta.some(Number.isNaN)) {
        console.error("Delta has NaNs detected during delta projection in convolutional layer");
        throw new Error(errFlags.NAN_ERR);
    }

    
    // 4. Cross-correlate with flipped kernels to get dL/da for the previous layer
    const result = ConvolveBackward(delta, targetShape, deltaShape, kernelShape, pointer, modelID, layerID);
    if (result.some(v => Number.isNaN(v))) {
        console.error("NaN detected during delta projection in convolutional layer");
        throw new Error(errFlags.NAN_ERR);
    }

    return result;
}

const applyDerivativeAndComputeGradients = (data) => {
    const layerData = data.layerData;
    const modelID = data.modelID;
    const incoming_delta = data.delta;
    const z = data.z_value;
    const wg = data.weightGrads;
    const bg = data.biasGrads;
    const activatedOutput = data.activatedOutput;
    const layerID = layerData.layerID;
    const kernelShape = layerData.weightShape;
    const inputShape = layerData.inputShape;
    const outputShape = layerData.outputShape; 
    
    const activationString = layerData.activation_function.name;

    const {delta, weightGrads, biasGrads} = fusedFinalDeltaProjector_ForConv(z, activatedOutput, incoming_delta, wg, bg, inputShape, outputShape, kernelShape, layerData.strides, activationString, modelID, layerID, !data.deltaAlreadyIncludesActivationDerivative);
    
    if (delta.some(v => Number.isNaN(v))) {
        console.error("NaN detected during derivative application in convolutional layer");
        throw new Error(errFlags.NAN_ERR);
    }
    
    if (weightGrads.some(v => Number.isNaN(v)) || biasGrads.some(v => Number.isNaN(v))) {
        console.error("NaN detected during gradient accumulation in convolutional layer");
        throw new Error(errFlags.NAN_ERR);
    }
    
    return {
        delta: delta,
        weightGrads: weightGrads,
        biasGrads: biasGrads
    };
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyDerivativeAndComputeGradients
}

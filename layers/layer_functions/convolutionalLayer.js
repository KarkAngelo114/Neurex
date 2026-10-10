const activation = require('../../core/bindings')
const { ConvolveForward, ConvolveBackward, element_wise_mul,AccumulateWeightAndBiasGradsForConv } = require("../../core/bindings");
const {  calculateTensorShape, createTensorBuffer } = require("../../utils/utils");



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
        throw new Error("ERR_NAN_DETECTED");
    }

    const totalSize = layerData.inputShape.reduce((acc, val) => acc * val, 1);

    if (input.length != totalSize) {
        console.error(`Input tensor doesn't match with the expected input tensor shape: Expected shape/size: ${[input_H, input_W, input_D]} or ${totalSize}. The size of the input entered is ${input.length}`);
        throw new Error("EERR_CONV_SHAPE_MISMATCH");
    }

    // 4. Perform the convolve operation using the shapes calculated in step 1
    const convolve_result = ConvolveForward(input, inputShape, outputShape, kernelShape, pointer, modelID, layerID);

    if (convolve_result.some(Number.isNaN)) {
        console.error("NaN detected after Convolve operation during feedforward in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }

    // 5. activate each depth input using the given activation function
    const activation_function = activation[layerData.activation_function.name];
    const outputs = activation_function(convolve_result, modelID, layerID);

    if (outputs.some(v => Number.isNaN(v))) {
        console.error("NaN detected after activation function during feedforward in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }

    layerData.cache = {
        layer_output: outputs,
    }

    return {
        outputs: outputs,
        z_values: convolve_result,
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
        throw new Error("ERR_NAN_DETECTED");
    }

    
    // 4. Cross-correlate with flipped kernels to get dL/da for the previous layer
    const result = ConvolveBackward(delta, targetShape, deltaShape, kernelShape, pointer, modelID, layerID);
    if (result.some(v => Number.isNaN(v))) {
        console.error("NaN detected during delta projection in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }

    return result;
}

const applyOwnDerivative = (data) => {
    const layerData = data.layerData;
    const delta = data.delta;
    const z = data.z_value;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const dActivation = activation.derivatives[layerData.activation_function.name];
    const storedOutput = layerData.cache.layer_output;
    const dAct = dActivation(z, storedOutput, modelID, layerID);

    if (dAct.some(Number.isNaN)) {
        console.error("Delta Derivative has NaNs detected after applying derivative activation in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }
    
    const result = element_wise_mul(dAct, delta, modelID, layerID);

    if (result.some(v => Number.isNaN(v))) {
        console.error("NaN detected after applying derivative activation in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }

    return result;
}

const gradientAccumulation = (data) => {
    const layerData = data.layerData;
    const deltas = data.deltas;
    const activation_outputs = data.activation_outputs;
    const weightGrads = data.weightGrads;
    const biasGrads = data.biasGrads;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const kernelShape = layerData.weightShape
    const inputShape = layerData.inputShape
    const outputShape = layerData.outputShape

    const {weightGrads: kernelWeightGrads, biasGrads: kernelBiasGrads} = AccumulateWeightAndBiasGradsForConv(activation_outputs, deltas, weightGrads, biasGrads, inputShape, outputShape, kernelShape, modelID, layerID);

    if (kernelWeightGrads.some(v => Number.isNaN(v))) {
        console.error("NaN detected after kernel weightGrads accumulation in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }

    if (kernelBiasGrads.some(Number.isNaN)) {
        console.error("NaN detected after kernel biasGrads accumulation in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }


    return {
        accumulatedWeightGrads: kernelWeightGrads,
        accumulatedBiasGrads: kernelBiasGrads
    }
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyOwnDerivative,
    gradientAccumulation
}

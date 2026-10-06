const activation = require('../../core/bindings')
const { applyPadding, Convolve, ConvolveDelta, element_wise_mul, Dilate_Input, DeltaMatMul, ComputeGradientForKernels, computeBiasGradsForConv } = require("../../core/bindings");
const {  calculateTensorShape, getPaddingSizes, createTensorBuffer } = require("../../utils/utils");



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

/**
 * Determeines what is the task the model is trained on
 * @param {Object} layerObject layer configuration object
 * @param {String} lossFunc loss function is used for training
 * @param {Float32Array} trainY target labels 
 * @returns {string} task type
 */
const determineInferenceType = (layerObject, lossFunc, trainY) => {
    throw new Error('Convolutional layer cannot be an output layer for now');
}

const feedforward = (data) => {
    const layerData = data.layerData;
    const strides = layerData.strides;
    const [f, kh, kw, kd] = layerData.weightShape;
    const [input_H, input_W, input_D] = layerData.inputShape; 
    const padding = layerData.padding;
    const input = data.input;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const totalSize = layerData.inputShape.reduce((acc, val) => acc * val, 1);

    if (input.length != totalSize) {
        console.error(`Input tensor doesn't match with the expected input tensor shape: Expected shape/size: ${[input_H, input_W, input_D]} or ${totalSize}. The size of the input entered is ${input.length}`);
        throw new Error("EERR_CONV_SHAPE_MISMATCH");
    }

    // 1. compute expected output tensor shape
    const { OutputHeight, OutputWidth } = calculateTensorShape(input_H, input_W, kh, kw, input_D, strides, padding);

    // 2. get padding sizes for each sides
    const {top, bottom, left, right} = getPaddingSizes(input_H, input_W, kh, kw, strides, padding);

    // 3. apply padding
    const {data: paddedTensor, shape} = applyPadding(input, input_H, input_W, input_D, top, bottom, left, right);

    // 4. Perform the convolve operation using the shapes calculated in step 1
    const convolve_result = Convolve(paddedTensor, strides, [OutputHeight, OutputWidth], [f, kh, kw, kd], [shape[0], shape[1]], pointer, modelID, layerID);

    if (convolve_result.some(Number.isNaN)) {
        console.error("NaN detected after Convolve operation during feedforward in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }

    // 5. activate each depth input using the given activation function
    const activation_function = activation[layerData.activation_function.name];
    const outputs = activation_function(convolve_result, modelID,layerID);

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
    const targetShape = data.inputShape;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const [Fn, KHn, KWn, KCn] = layerData.weightShape;
    const [oHn, oWn, oDn]= layerData.outputShape;
    const [oHprev, oWprev] = layerData.inputShape;
    const stridesN = layerData.strides;
    const paddingN = layerData.padding;

    // 1. Dilate the delta to undo the strides used in the forward pass
    const { data: dilated, dilatedHeight: dilatedH, dilatedWidth: dilatedW } = Dilate_Input(delta, [oHn, oWn, oDn], stridesN);

    // 2. Determine how much padding to add around the dilated delta so that the full-convolution with the flipped kernel lands on the correct shape.
    let pT, pB, pL, pR;
    if (paddingN === "valid") {
        // "valid" forward → "full" backward: pad K-1 on every side
        pT = pB = KHn - 1;
        pL = pR = KWn - 1;
    } else {
        // "same" forward: split K-1, then top up so the result is at least oHprev × oWprev
        pT = Math.floor((KHn - 1) / 2);  pB = (KHn - 1) - pT;
        pL = Math.floor((KWn - 1) / 2);  pR = (KWn - 1) - pL;

        const needH = oHprev + KHn - 1;   // ConvolveDelta needs Hp >= needH
        const needW = oWprev + KWn - 1;
        const haveH = dilatedH + pT + pB;
        const haveW = dilatedW + pL + pR;
        if (haveH < needH) pB += (needH - haveH);
        if (haveW < needW) pR += (needW - haveW);
    }

    // 3. Apply padding
    const { data: paddedInput, shape } = applyPadding(dilated, dilatedH, dilatedW, oDn, pT, pB, pL, pR);

    // 4. Cross-correlate with flipped kernels to get dL/da for the previous layer
    const result = ConvolveDelta(paddedInput, shape, [Fn, KHn, KWn, KCn], [oHprev, oWprev], pointer, 1, modelID, layerID);
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
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const dActivation = activation.derivatives[layerData.activation_function.name];
    const storedOutput = layerData.cache.layer_output;
    const dAct = dActivation(z, storedOutput, modelID, layerID);
    
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

    const [filters, kH, kW, inDepth] = layerData.weightShape
    const [inH, inW] = layerData.inputShape
    const [outH, outW] = layerData.outputShape


    const kernelWeightGrads = ComputeGradientForKernels(
        activation_outputs,
        deltas,
        weightGrads,
        [inH, inW, inDepth],
        [outH, outW, filters],
        [kH, kW],
        1,
        pointer, 
        modelID,
        layerID
    );

    if (kernelWeightGrads.some(v => Number.isNaN(v))) {
        console.error("NaN detected after kernel weightGrads accumulation in convolutional layer");
        throw new Error("ERR_NAN_DETECTED");
    }


    const kernelBiasGrads =  computeBiasGradsForConv(biasGrads, deltas, outH, outW, filters, pointer, modelID, layerID);

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

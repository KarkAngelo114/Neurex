const activation = require('../../core/bindings')
const { applyPadding, Convolve, ConvolveDelta, element_wise_mul, Dilate_Input, DeltaMatMul, ComputeGradientForKernels, computeBiasGradsForConv } = require("../../core/bindings");
const { XavierInitialization, calculateTensorShape, getPaddingSizes, createTensorBuffer } = require("../../utils/utils");



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

const feedforward = (input, current_layer, pointer, modelID) => {
    let [f, kh, kw, kd] = current_layer.weightShape;
    let [input_H, input_W, input_D] = current_layer.inputShape; 
    let padding = current_layer.padding;
    let strides = current_layer.strides;

    const totalSize = current_layer.inputShape.reduce((acc, val) => acc * val, 1);
    if (input.length != totalSize) throw new Error(`[CONV ERROR]------- Input tensor doesn't match with the expected input tensor shape: Expected shape/size: ${[input_H, input_W, input_D]} or ${totalSize}. The size of the input entered is ${input.length}`); 

    // 1. compute expected output tensor shape
    const { OutputHeight, OutputWidth } = calculateTensorShape(input_H, input_W, kh, kw, input_D, current_layer.strides, current_layer.padding);

    // 2. get padding sizes for each sides
    const {top, bottom, left, right} = getPaddingSizes(input_H, input_W, kh, kw, strides, padding);

    // 3. apply padding
    const {data, shape} = applyPadding(input, input_H, input_W, input_D, top, bottom, left, right);

    // 4. Perform the convolve operation using the shapes calculated in step 1
    const convolve_result = Convolve(data, current_layer.strides, [OutputHeight, OutputWidth], [f, kh, kw, kd], [shape[0], shape[1]], pointer, modelID);

    if (convolve_result.some(Number.isNaN)) throw new Error('NaN detected on convolve result');

    // 5. activate each depth input using the given activation function
    const activation_function = activation[current_layer.activation_function.name];
    const outputs = activation_function(convolve_result, pointer, modelID);

    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");

    current_layer.cache = {
        layer_output: outputs,
    }

    return {
        outputs: outputs,
        z_values: convolve_result,
        incrementor_value: 1
    };
}

const getOutputLayerDelta = (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
    throw new Error('Convolutional layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
}

const projectDeltaBackward = (delta, pointer, targetShape, layer_data, modelID) => {
    const [Fn, KHn, KWn, KCn] = layer_data.weightShape;
    const [oHn, oWn, oDn]     = layer_data.outputShape;
    const [oHprev, oWprev]    = targetShape;   // output shape of the layer before this one
    const stridesN             = layer_data.strides;
    const paddingN             = layer_data.padding;

    // 1. Dilate the delta to undo the strides used in the forward pass
    const { data: dilated, dilatedHeight: dilatedH, dilatedWidth: dilatedW } =
        Dilate_Input(delta, [oHn, oWn, oDn], stridesN);

    // 2. Determine how much padding to add around the dilated delta so that
    //    the full-convolution with the flipped kernel lands on the correct shape.
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
    const result = ConvolveDelta(paddedInput, shape, [Fn, KHn, KWn, KCn], [oHprev, oWprev], pointer, 1, modelID);
    if (result.some(v => Number.isNaN(v))) throw new Error("ConvolveDelta result has NaNs in projectDeltaBackward (convolutionalLayer)");

    return result;
}

const applyOwnDerivative = (delta, z, layer_data, pointer, modelID) => {
    const dActivation = activation.derivatives[layer_data.activation_function.name];
    const storedOutput = layer_data.cache.layer_output;
    const dAct = dActivation(z, storedOutput, pointer, modelID);
    
    const result = element_wise_mul(dAct, delta);
    if (result.some(v => Number.isNaN(v))) throw new Error("element_wise_mul result has NaNs in applyOwnDerivative (convolutionalLayer)");
    return result;
}


const computeWeightGradients = (activation_outputs, deltas, weightGrads, layer_data, pointer, modelID) => {
    const [filters, kH, kW, inDepth] = layer_data.weightShape
    const [inH, inW] = layer_data.inputShape
    const [outH, outW] = layer_data.outputShape


    const output = ComputeGradientForKernels(
        activation_outputs,
        deltas,
        weightGrads,
        [inH, inW, inDepth],
        [outH, outW, filters],
        [kH, kW],
        1,
        pointer, 
        modelID
    );

    if (output.some(Number.isNaN)) throw new Error(`Has NaNs after accumulation of kernel grads`);

    return output;
}

const computeBiasGradients = (biasgrads, deltas, layer_data, pointer, modelID) => {
    const [filters] = layer_data.weightShape;
    const [outH, outW] = layer_data.outputShape;
    return computeBiasGradsForConv(biasgrads, deltas, outH, outW, filters, pointer, modelID);
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyOwnDerivative,
    computeWeightGradients,
    computeBiasGradients
}

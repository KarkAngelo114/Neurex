const { red, reset } = require('../../color-code');
const activation = require('../../core/bindings');
const { applyPadding, element_wise_sub, scaleDiff, Convolve, ConvolveDelta, element_wise_mul, Dilate_Input, DeltaMatMul, ComputeGradientForKernels, computeBiasGradsForConv } = require("../../core/bindings");
const { XavierInitialization, calculateTransposedTensorShape, getTransposedPaddingSizes } = require('../../utils/utils');


/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, outputTensors: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>, isParametric: Boolean}}
 */
const initParams = (size, shape, layer_data) => {

    const currentInputShape = layer_data.inputShape.reduce((acc, current) => acc * current, 1);
    const incomingInputShape = shape.reduce((acc, current) => acc * current, 1);

    if (currentInputShape != incomingInputShape) {
        throw new Error(`[ERROR]------- Failed to initialized transpose convolution layer. Incoming shape must match current inputShape. Expected shape: ${layer_data.inputShape} or ${currentInputShape} | Incoming shape: ${shape} or ${incomingInputShape}`);
        process.exit(1);
    }
    
    const filters = layer_data.filters;
    const padding = layer_data.padding || "same";
    const [kh, kw] = layer_data.kernel_size || [3, 3];
    const [iH, iW, iD] = layer_data.inputShape || [28, 28, 1];
    const strides = layer_data.strides || 1;
    const TotalSize = filters * kh * kw * iD;

    const weights = new Float32Array(TotalSize);
    const biases = new Float32Array(filters);
    const weightGrads = new Float32Array(weights.length);
    const biasGrads =  new Float32Array(biases.length);

    const fanIn = kh * kw * iD;
    const fanOut = kh * kw * filters;
    const limit = XavierInitialization(fanIn, fanOut);

    // weights
    for (let i = 0; i < TotalSize; i++) {
        weights[i] =  (Math.random() * 2 - 1) * limit;
    }

    // biases
    for (let i = 0; i < filters; i++) {
        biases[i] =  (Math.random() * 2 - 1) * limit;
    }

    // calculate output shape
    const {OutputHeight, OutputWidth, CalculatedTensorShape} = calculateTransposedTensorShape(iH, iW, kh, kw, filters, strides, padding);

    // create allocated buffer (for GPU)
    const outputTensorTemplate = new Float32Array(CalculatedTensorShape);

    // output shape and weight shape
    const outputShape = [OutputHeight, OutputWidth, filters];
    const weightShape = [filters, kh, kw, iD];
                    
    return {
        updatedSize: CalculatedTensorShape,
        updatedShape: outputShape,
        weights: weights,
        biases: biases,
        weightGrads: weightGrads,
        biasGrads: biasGrads,
        outputTensors: outputTensorTemplate,
        inputShape: layer_data.inputShape,
        outputShape: outputShape,
        paramShape: weightShape,
        isParametric: true,
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

    if (lossFunc === "mae" || lossFunc === "mse") {
        return "regression";
    }

    if (lossFunc === "binary_cross_entropy") {
        return "binary_classification";
    }

    if (lossFunc === "categorical_cross_entropy" || lossFunc === "sparse_categorical_cross_entropy") {
        return "multi_class_classification";
    }

    //  if none satisfies the conditions above, throw an error
    throw new Error(`${red}[ERROR]------- Unknown loss function: ${lossFunc}${reset}`);
}

/**
 * The feedforward logic of this layer
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer, outputTemplatePointer) => {
    let [f, kh, kw, kd] = current_layer.weightShape;
    let [input_H, input_W, input_D] = current_layer.inputShape; 
    let [oH, oW, oD] = current_layer.outputShape; 
    let padding = current_layer.padding;
    let strides = current_layer.strides;
    let activationFunction = activation[current_layer.activation_function.name];

    // trans conv is just a inverse version of normal convolution where instead of shrinking the output by stride, the trans conv increase the output

    const { data: DilatedInput, dilatedHeight, dilatedWidth } = Dilate_Input(input, [input_H, input_W, input_D], strides);
    const { top, bottom, left, right } = getTransposedPaddingSizes(input_H, input_W, kh, kw, strides, padding);
    const { data: paddedInput, shape } = applyPadding(DilatedInput, dilatedHeight, dilatedWidth, kd, top, bottom, left, right);
    const convoleRes = Convolve(paddedInput, 1, [oH, oW, oD], [f, kh, kw, kd], [shape[0], shape[1]], pointer, outputTemplatePointer);
    if (convoleRes.some(v => Number.isNaN(v))) throw new Error("[TRANSCONV ERROR] Convolution output has Nans");

    // apply activation
    const output = activationFunction(convoleRes);
    if (output.some(v => Number.isNaN(v))) throw new Error("[TRANSCONV ERROR] output activation has Nans");

    return {
        outputs: output,
        z_values: convoleRes,
        incrementor_value: 1
    }
}

/**
 * 
 * @param {Float32Array} preds array of predicton outputs 
 * @param {Float32Array} actuals array of target labels 
 * @param {Array<Float32Array>} zs array of pre-activated values (zs)
 * @param {String} lossFunc loss function used in training
 * @param {String} tasktype task type the model is trained for 
 * @param {Object} layerObj layer config object of the last layer
 * @returns {Float32Array} the delta of the output layer
 */
const getOutputLayerDelta = (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
    let dActivation = activation.derivatives[layerObj.activation_function.name];
    let dOutputLayer = new Float32Array(preds.length); 

    if (tasktype === "binary_classification" || (tasktype === "multi_class_classification" && lossFunc === "categorical_cross_entropy")) {
        dOutputLayer = element_wise_sub(preds, actuals);
    }
    else if (tasktype === "multi_class_classification" && lossFunc === "sparse_categorical_cross_entropy") {
        dOutputLayer.set(preds);
        dOutputLayer[actuals[0]] -= 1;
                        
    }
    else if (tasktype === "regression") {
        if (preds.length != actuals.length) throw new Error("Predictions array is not equal to actuals array");

        const lastLayerZs = zs[zs.length - 1]; 
        const dAct = dActivation(lastLayerZs); 

        dOutputLayer = scaleDiff(preds, actuals, dAct);

        if (dOutputLayer.some(v => Number.isNaN(v))) throw new Error("Delta of the output layer has NaNs"); 

    }

    return dOutputLayer;
}

/**
 * Projects the incoming delta backward through THIS conv layer's own kernels.
 * Called on the *next* layer (in feedforward direction) from the core backprop loop.
 * No branching on layer type — each layer implements this for itself.
 *
 * Math: we dilate the incoming delta (to undo strides), pad it (to undo the
 * original padding mode), then cross-correlate with the flipped kernels to
 * recover the gradient w.r.t. the previous layer's output.
 *
 * @param {Float32Array} delta - incoming delta from the layer ahead (in backprop direction)
 * @param {Number} pointer - weight pointer for THIS conv layer
 * @param {Array<Number>} targetShape - outputShape of the layer that will *receive* the projected delta
 * @param {Object} layer_data - THIS conv layer's own configuration (weightShape, outputShape, strides, padding)
 * @returns {Float32Array} projected delta (dL/da for the previous layer's activations)
 */
const projectDeltaBackward = (delta, pointer, targetShape, layer_data) => {
    const [Fn, KHn, KWn, KCn] = layer_data.weightShape;
    const [oHn, oWn, oDn] = layer_data.outputShape;
    
    const [oHprev, oWprev] = layer_data.inputShape; 
    
    const stridesN = layer_data.strides;
    const paddingN = layer_data.padding;

    let pT, pB, pL, pR;
    if (paddingN === "valid") {
        pT = pB = 0;   // was KHn - 1
        pL = pR = 0;   // was KWn - 1
    } else {
        // "same" — existing logic is correct for stride=1
        pT = Math.floor((KHn - 1) / 2);  pB = (KHn - 1) - pT;
        pL = Math.floor((KWn - 1) / 2);  pR = (KWn - 1) - pL;
        // keep the needH/needW clamp as-is
    }

    const { data: paddedInput, shape } = applyPadding(delta, oHn, oWn, oDn, pT, pB, pL, pR);

    const result = ConvolveDelta(paddedInput, shape, [Fn, KHn, KWn, KCn], [oHprev, oWprev], pointer, 1);
    if (result.some(v => Number.isNaN(v))) throw new Error("ConvolveDelta result has NaNs in projectDeltaBackward (trans conv)");
    
    return result;
}

/**
 * Applies this conv layer's own activation derivative to the projected delta.
 * Called on the *current* layer from the core backprop loop.
 *
 * @param {Float32Array} delta - projected delta (output of next_layer.projectDeltaBackward)
 * @param {Float32Array} z - pre-activation values (z) for this layer
 * @param {Object} layer_data - this layer's own configuration
 * @returns {Float32Array} delta for the layer before this one
 */
const applyOwnDerivative = (delta, z, layer_data) => {
    const dActivation = activation.derivatives[layer_data.activation_function.name];
    const result = element_wise_mul(dActivation(z), delta);
    if (result.some(v => Number.isNaN(v))) throw new Error("element_wise_mul result has NaNs in applyOwnDerivative (trans conv)");

    return result;
}

const accumulateKernelGrads = (activation_outputs, deltas, weightGrads, layer_data) => {
    const [filters, kH, kW, inDepth] = layer_data.weightShape
    const [inH, inW, inD] = layer_data.inputShape;
    const [outH, outW] = layer_data.outputShape;
    const strides = layer_data.strides;

    const {data: dilated, dilatedHeight, dilatedWidth} = Dilate_Input(activation_outputs, [inH, inW, inD], strides);

    const output = ComputeGradientForKernels(
        dilated,
        deltas,
        weightGrads,
        [dilatedHeight, dilatedWidth, inD],
        [outH, outW, filters],
        [kH, kW]
    );

    if (output.some(Number.isNaN)) throw new Error(`Has NaNs after accumulation of kernel grads (trans conv)`);

    return output;
}

/**
 * 
 * @param {Float32Array} biasgrads initially zeroed gradient accumulators 
 * @param {Float32Array} deltas all outputs during backpropagation
 * @param {Object} layer_data layer configuration data
 * @returns {Float32Array} Float32Array accumulated gradients
 */
const accumulateBiasGradients = (biasgrads, deltas, layer_data) => {
    const [filters] = layer_data.weightShape;
    const [outH, outW] = layer_data.outputShape;
    const output =  computeBiasGradsForConv(biasgrads, deltas, outH, outW, filters);
    
    if (output.some(v => Number.isNaN(v))) throw new Error("bias gradient accumulation result has NaNs in accumulateBiasGradients (trans conv)");
    return output;
}


module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyOwnDerivative,
    accumulateKernelGrads,
    accumulateBiasGradients
}
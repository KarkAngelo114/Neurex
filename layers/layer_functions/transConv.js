const { red, reset } = require('../../color-code');
const activation = require('../../core/bindings');
const { transConv, computeBiasGradsForConv, scaleDiff, transConvBackward, element_wise_mul} = require("../../core/bindings");
const { XavierInitialization, calculateTransposedTensorShape, getTransposedPaddingSizes } = require('../../utils/utils');


/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, outputTensors: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>}}
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
    
    const inputShape = current_layer.inputShape; // [iH, iW, iD]
    const outputShape = current_layer.outputShape; // [oH, oW, oD]
    const weightShape = current_layer.weightShape; // [f, kh, kw, d]
    const kernelSize = current_layer.kernel_size; // [kh, kw]
    const strides = current_layer.strides;
    const filters = current_layer.filters;
    const activation_function = activation[current_layer.activation_function.name];

    const transConvOutput = transConv(input, inputShape, outputShape, strides, filters, kernelSize, weightShape, pointer, outputTemplatePointer);
    if (transConvOutput.some(v => Number.isNaN(v))) throw new Error("[Trans Conv Error] output array has NaNs after trans conv Ops");

    const output = activation_function(transConvOutput);
    if (output.some(v => Number.isNaN(v))) throw new Error("[Trans Conv Error] output array has NaNs after applying activation");

    return {
        outputs: output,
        z_values: transConvOutput,
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
 *
 * @param {Float32Array} delta - incoming delta from the layer ahead (in backprop direction)
 * @param {Number} pointer - weight pointer for THIS conv layer
 * @param {Array<Number>} targetShape - outputShape of the layer that will *receive* the projected delta
 * @param {Object} layer_data - THIS conv layer's own configuration (weightShape, outputShape, strides, padding)
 * @returns {Float32Array} projected delta (dL/da for the previous layer's activations)
 */
const projectDeltaBackward = (delta, pointer, targetShape, layer_data) => {
    const inputShape = layer_data.inputShape;
    const outputShape = layer_data.outputShape;
    const weightShape = layer_data.weightShape;
    const kernelSize = layer_data.kernel_size;
    const strides = layer_data.strides;
    const filters = layer_data.filters;

    const result = transConvBackward(delta, inputShape, outputShape, strides, filters, kernelSize, weightShape, pointer);
    if (result.some(v => Number.isNaN(v))) throw new Error("[Trans Conv Delta Projection Error] output array has NaNs after transConvBackward() Ops");
    
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
    // TODO
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
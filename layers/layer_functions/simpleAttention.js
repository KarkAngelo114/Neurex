const activation = require('../../core/bindings');
const { CoreAttention } = require('../../core/bindings');
const { XavierInitialization, concatenateFloat32Array } = require('../../utils');

/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>}}
 */
const initParams = (size, shape, layer_data) => {
    // assume that next to embedding layer is simpleAttention(),
    // the incoming shape is [1, 1, embedDim, seqLen]
    const embeddingDim = shape[2];
    const useBias = layer_data.useBias;

    const Q_weights = new Float32Array(embeddingDim * embeddingDim);
    const K_weights = new Float32Array(embeddingDim * embeddingDim);
    const V_weights = new Float32Array(embeddingDim * embeddingDim);

    const limit = XavierInitialization(embeddingDim, embeddingDim);
    for (let i = 0; i < Q_weights.length; i++) {
        Q_weights[i] = (Math.random() * 2 - 1) * limit;
        K_weights[i] = (Math.random() * 2 - 1) * limit;
        V_weights[i] = (Math.random() * 2 - 1) * limit;
    }

    const biases = new Float32Array(embeddingDim * 3);
    if (useBias) {
        for (let i = 0; i < biases.length; i++) {
            biases[i] = (Math.random() * 2 - 1) * limit;
        }
    }

    const weights = concatenateFloat32Array([Q_weights, K_weights, V_weights]);
    const weightShape = [embeddingDim, embeddingDim * 3]; // combined QKV projection

    layer_data.dkRoot = Math.sqrt(embeddingDim);
    layer_data.embedDim = embeddingDim;
    layer_data.seqLen = shape[3];

    return {
        updatedSize: embeddingDim,
        updatedShape: [1, 1, embeddingDim, shape[3]], // seqLen unchanged, embedDim unchanged
        weights,
        biases,
        weightGrads: new Float32Array(weights.length),
        biasGrads: new Float32Array(biases.length),
        inputShape: shape,
        outputShape: [1, 1, embeddingDim, shape[3]],
        paramShape: weightShape,
    };
}

/**
 * Determeines what is the task the model is trained on
 * @param {Object} layerObject layer configuration object
 * @param {String} lossFunc loss function is used for training
 * @param {Float32Array} trainY target labels 
 * @returns {string} task type
 */
const determineInferenceType = (layerObject, lossFunc, trainY) => {
    throw new Error('simple attention layer cannot be an output layer for now');
}

/**
 * The feedforward logic of this layer
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer) => {
    const output = CoreAttention(input, current_layer, pointer);

    if (output.some(v => Number.isNaN(v))) throw new Error("[ERROR]---- output array has NaNs (Simple Attention during feed forward)");

    current_layer.cache = {
        softmax_output: output
    }


    return {
        outputs: output,
        z_values: output,
        incrementor_value: 1
    };
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

    return [];
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
    if (result.some(v => Number.isNaN(v))) throw new Error("element_wise_mul result has NaNs in applyOwnDerivative (convolutionalLayer)");
    return result;
}

/**
 * 
 * @param {Float32Array} activation_outputs all outputs during feedforward
 * @param {Float32Array} deltas all outputs during backpropagation 
 * @param {Float32Array} weightGrads initially zeroed accumulators
 * @param {Object} layer_data layer configuration data
 * @returns {Float32Array} Float32Array accumulated gradients
 */
const accumulateWeightGradients = (activation_outputs, deltas, weightGrads, layer_data) => {
    return [];
}

/**
 * 
 * @param {Float32Array} biasgrads initially zeroed gradient accumulators 
 * @param {Float32Array} deltas all outputs during backpropagation
 * @param {Object} layer_data layer configuration data
 * @returns {Float32Array} Float32Array accumulated gradients
 */
const accumulateBiasGradients = (biasgrads, deltas, layer_data) => {
    return [];
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyOwnDerivative,
    accumulateWeightGradients,
    accumulateBiasGradients
}

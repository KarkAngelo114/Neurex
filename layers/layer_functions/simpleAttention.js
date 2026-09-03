const activation = require('../../core/bindings');
const { CoreAttention, CoreAttentionBackward, computeBiasGradsForConnected_Layer, computeWeightGradientsForWeightsInConnectedLayer } = require('../../core/bindings');
const { XavierInitialization, concatenateFloat32Array, unpackQKVO } = require('../../utils');

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
        updatedSize: embeddingDim * shape[3],
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
 * @param {String} modelID model ID
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer, modelID) => {
    const output = CoreAttention(input, current_layer, pointer, modelID);

    if (output.some(v => Number.isNaN(v))) throw new Error("[ERROR]---- output array has NaNs (Simple Attention during feed forward)");

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
}

/**
 *
 * @param {Float32Array} delta - incoming delta from the layer ahead (in backprop direction)
 * @param {Number} pointer - weight pointer
 * @param {Array<Number>} targetShape - outputShape of the layer that will *receive* the projected delta
 * @param {Object} layer_data - layer data
 * @param {String} modelID model ID
 * @returns {Float32Array} projected delta (dL/da for the previous layer's activations)
 */
const projectDeltaBackward = (delta, pointer, targetShape, layer_data, modelID) => {

    const output = CoreAttentionBackward(delta, layer_data, pointer, modelID);
    if (output.some(v => Number.isNaN(v))) throw new Error("[ERROR]---- output array has NaNs (Simple Attention during projecting delta backward)");
    return output;
}

/**
 * @param {Float32Array} delta - projected delta (output of next_layer.projectDeltaBackward)
 * @param {Float32Array} z - pre-activation values (z) for this layer
 * @param {Object} layer_data - this layer's own configuration
 * @returns {Float32Array} delta for the layer before this one
 */
const applyOwnDerivative = (delta) => {
    return delta;
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
    const { embedDim, seqLen, cache } = layer_data;
    const { dQ, dK, dV } = cache;
    const { Q_weightGrads: QwGrads, K_weightGrads: KwGrads, V_weightGrads: VwGrads } = unpackQKVO(null, null, weightGrads, null, embedDim);

    let QwG;
    let KwG;
    let VwG;

    for (let t = 0; t < seqLen; t++) {
        const Xrow  = activation_outputs.subarray(t * embedDim, (t + 1) * embedDim);
        const dQrow = dQ.subarray(t * embedDim, (t + 1) * embedDim);
        const dKrow = dK.subarray(t * embedDim, (t + 1) * embedDim);
        const dVrow = dV.subarray(t * embedDim, (t + 1) * embedDim);

        QwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dQrow, QwGrads, embedDim, embedDim);
        KwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dKrow, KwGrads, embedDim, embedDim);
        VwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dVrow, VwGrads, embedDim, embedDim);
    }

    return concatenateFloat32Array([QwG, KwG, VwG]);
}

/**
 * 
 * @param {Float32Array} biasgrads initially zeroed gradient accumulators 
 * @param {Float32Array} deltas all outputs during backpropagation
 * @param {Object} layer_data layer configuration data
 * @returns {Float32Array} Float32Array accumulated gradients
 */
const accumulateBiasGradients = (biasGrads, deltas, layer_data) => {
    const { embedDim, seqLen, cache } = layer_data;
    const { dQ, dK, dV } = cache;
    const { Q_biasGrads: QbGrads, K_biasGrads: KbGrads, V_biasGrads: VbGrads } = unpackQKVO(null, null, null, biasGrads, embedDim);
    
    let QbG;
    let KbG;
    let VbG;

    for (let t = 0; t < seqLen; t++) {
        QbG = computeBiasGradsForConnected_Layer(QbGrads, dQ.subarray(t * embedDim, (t + 1) * embedDim));
        KbG = computeBiasGradsForConnected_Layer(KbGrads, dK.subarray(t * embedDim, (t + 1) * embedDim));
        VbG = computeBiasGradsForConnected_Layer(VbGrads, dV.subarray(t * embedDim, (t + 1) * embedDim));
    }
    
    return concatenateFloat32Array([QbG, KbG, VbG]);
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

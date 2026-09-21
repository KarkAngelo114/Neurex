const { CoreMultiHeadAttention, CoreMultiHeadAttentionBackward, accumulateAttentionWeightsGradients, accumulateAttentionBiasGrads } = require('../../core/bindings/entry');;
const { createTensorBuffer, concatenateFloat32Array } = require('../../utils/utils');

/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>}}
 */
const initParams = (size, shape, layer_data) => {
    try {
        // assume that next to embedding layer is multiHeadAttention(),
        // the incoming shape is [1, 1, embedDim, seqLen]
        const embeddingDim = shape[2];
        const useBias = layer_data.useBias;
        const numHeads = layer_data.numHeads;
        const headDim = embeddingDim / numHeads;

        // in MHA, embedding dim must strictly be divisible by num_heads in order to have equal number of sets
        if (embeddingDim % numHeads != 0) {
            throw new Error(`[MULTI-HEAD ATTENTION ERROR]------- embeddingDim is not divisible to numHeads. Embedding dim: ${embeddingDim} | Num heads: ${numHeads}`);
        }

        const parameterOptions = {
            min: embeddingDim,
            max: embeddingDim,
            prefilledWith: "xavier"
        };
        const Q_weights = createTensorBuffer([embeddingDim, embeddingDim], parameterOptions).data;
        const K_weights = createTensorBuffer([embeddingDim, embeddingDim], parameterOptions).data;
        const V_weights = createTensorBuffer([embeddingDim, embeddingDim], parameterOptions).data;
        const O_weights = createTensorBuffer([embeddingDim, embeddingDim], parameterOptions).data;
        const biases = useBias
            ? createTensorBuffer([embeddingDim * 4], parameterOptions).data
            : new Float32Array(embeddingDim * 4);

        const weights = concatenateFloat32Array([Q_weights, K_weights, V_weights, O_weights]);
        const weightShape = [embeddingDim, embeddingDim * 4]; // combined QKVO projection

        layer_data.dkRoot = Math.sqrt(headDim);
        layer_data.embedDim = embeddingDim;
        layer_data.seqLen = shape[3] || 1;
        layer_data.headDim = headDim;

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
    catch (e) {
        console.error(e);
        process.exit(1);
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
    throw new Error('multi-head attention layer cannot be an output layer for now');
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

    const {embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking} = current_layer;

    const {Q, K, V, mhaOutput, S_perHead, finalOutput} = CoreMultiHeadAttention(input,  embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking, pointer, modelID);

    current_layer.cache = {
        X: input,
        Q: Q, 
        K: K, 
        V: V,
        mhaOutput: mhaOutput,
        S_perHead: S_perHead
    };

    if (finalOutput.some(v => Number.isNaN(v))) throw new Error("[ERROR]---- output array has NaNs (Multi-Head Attention during feed forward)");

    return {
        outputs: finalOutput,
        z_values: finalOutput,
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
 * @param {Number} pointer - weight pointer for layer
 * @param {Array<Number>} targetShape - outputShape of the layer that will *receive* the projected delta
 * @param {Object} layer_data - layer data
 * @param {String} modelID model ID
 * @returns {Float32Array} projected delta (dL/da for the previous layer's activations)
 */
const projectDeltaBackward = (delta, pointer, targetShape, layer_data, modelID) => {

    const {cache, embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking} = layer_data;
    const { Q, K, V, S_perHead } = cache;
    const {dQ, dK, dV, dMhaOutput, dX} = CoreMultiHeadAttentionBackward(delta, Q, K, V, S_perHead, embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking, pointer, modelID);

    layer_data.cache = {
        ...cache,
        dQ: dQ,
        dK: dK,
        dV: dV,
        dMhaOutput: dMhaOutput,
        dX: dX
    };

    if (dX.some(v => Number.isNaN(v))) throw new Error("[ERROR]---- output array has NaNs (Multi-Head Attention during projecting delta backward)");
    return dX;
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
    const { dQ, dK, dV, dMhaOutput, mhaOutput } = cache; 

    const output = accumulateAttentionWeightsGradients(dQ, dK, dV, dMhaOutput, mhaOutput, activation_outputs, weightGrads, embedDim, seqLen);
    if (output.some(v => Number.isNaN(v))) throw new Error("[ERROR] output array has NaNs (Multi-Head Attention during weight gradient accumulation)");

    return output;
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
    const { dQ, dK, dV, dMhaOutput } = cache;

    const output = accumulateAttentionBiasGrads(dQ, dK, dV, dMhaOutput, biasGrads, embedDim, seqLen);

    if (output.some(v => Number.isNaN(v))) throw new Error("[ERROR] output array has NaNs (Multi-Head Attention during bias gradient accumulation)");
    return output;

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

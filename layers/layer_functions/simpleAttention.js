const { CoreSimpleAttention, CoreSimpleAttentionBackward, accumulateSimpleAttentionWeightGrads, accumulateSimpleAttentionBiasGrads } = require('../../core/bindings');
const { createTensorBuffer, concatenateFloat32Array } = require('../../utils');
const err_flags = require('../../err_flags');

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

    const parameterOptions = {
        min: embeddingDim,
        max: embeddingDim,
        prefilledWith: "xavier"
    };
    const Q_weights = createTensorBuffer([embeddingDim, embeddingDim], parameterOptions).data;
    const K_weights = createTensorBuffer([embeddingDim, embeddingDim], parameterOptions).data;
    const V_weights = createTensorBuffer([embeddingDim, embeddingDim], parameterOptions).data;
    const biases = useBias
        ? createTensorBuffer([embeddingDim * 3], parameterOptions).data
        : new Float32Array(embeddingDim * 3);

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

const determineInferenceType = () => {
    throw new Error('simple attention layer cannot be an output layer for now');
}

const feedforward = (data) => {
    const layerData = data.layerData;
    const input = data.input;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const { embedDim, dkRoot, seqLen, useCausalMasking } = layerData;

    const {X, Q, K, V, S, output} = CoreSimpleAttention(input, embedDim, seqLen, dkRoot, useCausalMasking, pointer, modelID, layerID);

    layerData.cache = {
        X: X,
        Q: Q, 
        K: K, 
        V: V,
        S: S
    };

    if (output.some(v => Number.isNaN(v))) throw new Error("[ERROR]---- output array has NaNs (Simple Attention during feed forward)");

    return {
        outputs: output,
        z_values: output,
        incrementor_value: 1
    };
}

const getOutputLayerDelta = () => {
    throw new Error('simple attention layer cannot be an output layer for now');
}

const projectDeltaBackward = (data) => {
    return data.delta;
}

const applyDerivativeAndComputeGradients = (data) => {
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const incoming_delta = data.delta;
    const weight_grads = data.weightGrads;
    const bias_grads = data.biasGrads;
    const activatedOutput = data.activatedOutput;

    const layerID = layerData.layerID;

    const {cache, embedDim, seqLen, dkRoot, useCausalMasking} = layerData;
    const {Q, K, V, S } = cache;
    const {delta, weightGrads, biasGrads} = CoreSimpleAttentionBackward(incoming_delta, Q, K, V, S, weight_grads, bias_grads, activatedOutput, embedDim, seqLen, dkRoot, useCausalMasking, pointer, modelID, layerID);

    if (delta.some(v => Number.isNaN(v))) {
        console.error("[ERROR] output array has NaNs (Simple Attention during projecting delta backward)");
        throw new Error(err_flags.NAN_ERR);
    }

    if (weightGrads.some(v => Number.isNaN(v)) || biasGrads.some(v => Number.isNaN(v))) {
        console.error("[ERROR] gradients has NaNs in Simple Attention layer after gradient accumulation");
        throw new Error(err_flags.NAN_ERR);
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
    applyDerivativeAndComputeGradients,
}

const { CoreMultiHeadAttention, CoreMultiHeadAttentionBackward} = require('../../core/bindings/entry');
const { createTensorBuffer, concatenateFloat32Array } = require('../../utils/utils');
const errFlags = require('../../err_flags');

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

const determineInferenceType = () => {
    throw new Error('multi-head attention layer cannot be an output layer for now');
}

const feedforward = (data) => {

    const layerData = data.layerData;
    const modelID = data.modelID;
    const pointer = data.pointer;
    const input = data.input;
    const layerID = layerData.layerID;

    const {embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking} = layerData;

    const {Q, K, V, mhaOutput, S_perHead, finalOutput} = CoreMultiHeadAttention(input,  embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking, pointer, modelID, layerID);

    layerData.cache = {
        X: input,
        Q: Q, 
        K: K, 
        V: V,
        mhaOutput: mhaOutput,
        S_perHead: S_perHead
    };

    if (finalOutput.some(v => Number.isNaN(v))) {
        console.error("[ERROR] output array has NaNs (Multi-Head Attention during feed forward)");
        throw new Error(errFlags.NAN_ERR);
    }

    return {
        outputs: finalOutput,
        z_values: finalOutput,
        incrementor_value: 1
    };
}

const getOutputLayerDelta = () => {
    throw new Error('multi-head attention layer cannot be an output layer for now');
}

const projectDeltaBackward = (data) => {
    return data.delta;
}

const applyDerivativeAndComputeGradients = (data) => {
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const incoming_delta = data.delta;
    const layerID = layerData.layerID;
    const weight_grads = data.weightGrads;
    const bias_grads = data.biasGrads;
    const activatedOutput = data.activatedOutput;

    const {cache, embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking} = layerData;
    const { Q, K, V, S_perHead, mhaOutput } = cache;

    const {delta, weightGrads, biasGrads} = CoreMultiHeadAttentionBackward(incoming_delta, Q, K, V, S_perHead, mhaOutput, activatedOutput, weight_grads, bias_grads, embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking, pointer, modelID, layerID);

    
    return {
        delta: delta,
        weightGrads: weightGrads,
        biasGrads: biasGrads
    }
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyDerivativeAndComputeGradients,
}

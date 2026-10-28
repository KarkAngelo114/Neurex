const { computeLayerNorm, computeLayerNormBackward } = require("../../core/bindings/entry");
const { createTensorBuffer } = require("../../utils/utils");
const errFlags = require('../../err_flags');

const initParams = (size, shape, layer_data) => {
    // gamma initialized to 1s, beta initialized to 0s

    let gamma = createTensorBuffer([size], {prefilledWith: "ones"}).data;
    let beta = createTensorBuffer([size], {prefilledWith: "zeroes"}).data;
    let gammaGrads = createTensorBuffer([size], {prefilledWith: "zeroes"}).data;
    let betaGrads = createTensorBuffer([size], {prefilledWith: "zeroes"}).data;

    return {
        updatedSize: size,
        updatedShape: shape,
        weights: gamma,
        biases: beta,
        weightGrads: gammaGrads,
        biasGrads: betaGrads,
        inputShape: shape,
        outputShape: shape,
        paramShape: [size],
    };
};

const determineInferenceType = () => {
    throw new Error("[ERROR] LayerNorm cannot be used as an output layer.");
}

const feedforward = (data) => {
    const input = data.input;
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const eps = layerData.eps || 1e-5;
    const D = input.length;

    const outputs = computeLayerNorm(input, D, eps, pointer, modelID, layerID);

    if (outputs.some(v => Number.isNaN(v))) {
        console.error("NaN detected after normalization operation on layerNorm");
        throw new Error(errFlags.NAN_ERR);
    }

    layerData.cache = {
        X: input
    }

    return { 
        outputs: outputs, 
        z_values: outputs, 
        incrementor_value: 1 
    };
};

const getOutputLayerDelta = () => {
    throw new Error("[ERROR] LayerNorm cannot be used as an output layer.");
}

const applyDerivativeAndComputeGradients = (data) => {
    const incoming_delta = data.delta;
    const pointer = data.pointer;
    const weight_grads = data.weightGrads;
    const bias_grads = data.biasGrads;
    const layerData = data.layerData;
    const eps = layerData.eps || 1e-5;
    const X = layerData.cache.X;
    const size = incoming_delta.length;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const { delta, weightGrads, biasGrads } = computeLayerNormBackward(incoming_delta, X, weight_grads, bias_grads, size, eps, pointer, modelID, layerID);

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
    applyDerivativeAndComputeGradients,
}
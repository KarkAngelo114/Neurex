const { computeLayerNorm, accumulateGammaGrads: accumulateGammaGradsFunc, accumulateBetaGrads: accumulateBetaGradsFunc, computeLayerNormBackward } = require("../../core/bindings/entry");
const { createTensorBuffer } = require("../../utils/utils");

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

const feedforward = (input, current_layer, pointer, modelID) => {
    const eps = current_layer.eps || 1e-5;
    const D = input.length;

    const outputs = computeLayerNorm(input, D, eps, pointer, modelID);

    current_layer.cache = {
        X: input
    }

    return { outputs, z_values: outputs, incrementor_value: 1 };
};

const getOutputLayerDelta = () => {
    throw new Error("[ERROR] LayerNorm cannot be used as an output layer.");
}

const projectDeltaBackward = (delta, pointer, targetShape, layer_data, modelID) => {
    const X = layer_data.cache.X;
    const size = delta.length;

    const { dX, dGamma, dBeta } = computeLayerNormBackward(delta, X, size, pointer, modelID);

    layer_data.cache.dGamma = dGamma;
    layer_data.cache.dBeta = dBeta;

    return dX;
};

const accumulateGammaGrads = (a_prev, delta, gammaGrads, pointer, modelID, layer_data) => {
    const dGamma = layer_data.cache.dGamma;
    return accumulateGammaGradsFunc(gammaGrads, dGamma, pointer, modelID);
};

const accumulateBetaGrads = (betaGrads, delta, pointer, modelID, layer_data) => {
    const dBeta = layer_data.cache.dBeta;
    return accumulateBetaGradsFunc(betaGrads, dBeta, pointer, modelID);
};

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyOwnDerivative,
    accumulateGammaGrads,
    accumulateBetaGrads,
}
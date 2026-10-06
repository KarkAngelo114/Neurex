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
        throw new Error("ERR_NAN_DETECTED");
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

const applyOwnDerivative = (data) => {
    const delta = data.delta;
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;
    const eps = layerData.eps || 1e-5;

    const X = layerData.cache.X;
    const size = delta.length;

    const { dX, dGamma, dBeta } = computeLayerNormBackward(delta, X, size, eps, pointer, modelID, layerID);

    layerData.cache.dGamma = dGamma;
    layerData.cache.dBeta = dBeta;

    if (dX.some(v => Number.isNaN(v))) {
        console.error("layerNorm has NaNs after delta projection");
        throw new Error("ERR_NAN_DETECTED");
    }
    
    return dX;
};

const gradientAccumulation = (data) => {
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;
    const gammaGrads = data.weightGrads;
    const betaGrads = data.biasGrads;

    const dGamma = layerData.cache.dGamma;
    const accumulatedGammaGrads = accumulateGammaGradsFunc(gammaGrads, dGamma, pointer, modelID, layerID);

    if (accumulatedGammaGrads.some(v => Number.isNaN(v))) {
        console.error("NaN detected after accumulating gamma grads");
        throw new Error("ERR_NAN_DETECTED");
    }
    
    const dBeta = layerData.cache.dBeta;
    const accumulatedBetaGrads = accumulateBetaGradsFunc(betaGrads, dBeta, pointer, modelID, layerID);

    if (accumulatedBetaGrads.some(v => Number.isNaN(v))) {
        console.error("NaN detected after accumulating beta grads");
        throw new Error("ERR_NAN_DETECTED");
    }

    return {
        accumulatedWeightGrads: accumulatedGammaGrads,
        accumulatedBiasGrads: accumulatedBetaGrads
    }
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    applyOwnDerivative,
    gradientAccumulation
}
const { computeLayerNorm, computeLayerNormBackward } = require("../../core/bindings/entry");
const { createTensorBuffer } = require("../../utils/utils");
const errFlags = require('../../err_flags');

const initParams = (size, shape, layer_data) => {
    // gamma initialized to 1s, beta initialized to 0s

    let gamma = createTensorBuffer([size], {prefilledWith: "ones"}).data;
    let beta = createTensorBuffer([size], {prefilledWith: "zeroes"}).data;

    layer_data.weightShape = [size];

    layer_data.weights = gamma;
    layer_data.biases = beta;

    return {
        updatedSize: size,
        updatedShape: shape,
        weights: gamma,
        biases: beta,
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
    const weights = layerData.weights;
    const biases = layerData.biases;

    const eps = layerData.eps || 1e-5;
    const D = input.length;

    const outputs = computeLayerNorm(input, weights, biases, D, eps, pointer, modelID, layerID);

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
    const layerData = data.layerData;
    const incoming_delta = data.delta;
    const pointer = data.pointer;
    const eps = layerData.eps || 1e-5;
    const X = layerData.cache.X;
    const size = incoming_delta.length;
    const modelID = data.modelID;
    const layerID = layerData.layerID;
    const weightShape = layerData.weightShape;
    const weight_grads = layerData.weightGrads || createTensorBuffer(weightShape, {prefilledWith:"zeroes"}).data;
    const bias_grads = layerData.biasGrads || createTensorBuffer(weightShape, {prefilledWith:"zeroes"}).data;
    const weights = layerData.weights;

    const { delta, weightGrads, biasGrads } = computeLayerNormBackward(incoming_delta, X, weights, weight_grads, bias_grads, size, eps, pointer, modelID, layerID);

    layerData.weightGrads = weightGrads;
    layerData.biasGrads = biasGrads;

    return delta;
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    applyDerivativeAndComputeGradients,
}
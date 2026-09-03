const { computeLayerNorm, accumulate_element_wise_mul, computeBiasGradsForConnected_Layer } = require("../../core/bindings/entry");

const initParams = (size, shape, layer_data) => {
    // gamma initialized to 1s, beta initialized to 0s
    const gamma = new Float32Array(size).fill(1.0);
    const beta = new Float32Array(size).fill(0.0);
    const gammaGrads = new Float32Array(size).fill(0.0);
    const betaGrads = new Float32Array(size).fill(0.0);

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

    return { outputs, z_values: outputs, incrementor_value: 1 };
};

const getOutputLayerDelta = () => {
    throw new Error("[ERROR] LayerNorm cannot be used as an output layer.");
}

const projectDeltaBackward = (delta, pointer, targetShape, layer_data) => {
    // Pass spatial or upstream gradients directly backward through scale/shift derivative
    return delta;
}

const applyOwnDerivative = (delta, z, layer_data)  => {
    return delta;
}

const accumulateGammaGrads = (a_prev, delta, gammaGrads, layer_data) => {
    return accumulate_element_wise_mul(a_prev, delta, gammaGrads);
}

const accumulateBetaGrads = (betaGrads, delta) => {
    
    const output =  computeBiasGradsForConnected_Layer(betaGrads, delta);

    return output;
}

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
const { computeLayerNorm } = require("../../core/bindings/entry");

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

const feedforward = (input, current_layer, pointer) => {
    const eps = current_layer.eps || 1e-5;
    const D = input.length;

    const outputs = computeLayerNorm(input, D, eps, pointer);

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
    for (let i = 0; i < delta.length; i++) {
        gammaGrads[i] += delta[i] * a_prev[i];
    }
    return gammaGrads;
}

const accumulateBetaGrads = (betaGrads, delta) => {
    for (let i = 0; i < delta.length; i++) {
        betaGrads[i] += delta[i];
    }
    
    return betaGrads;
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
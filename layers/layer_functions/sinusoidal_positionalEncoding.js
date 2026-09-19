const { sinusoidalPE } = require("../../core/bindings/entry");

const initParams = (size, shape, layer_data) => {
    if (!Array.isArray(shape) || shape.length < 4) {
        throw new Error(
            `[SINUSOIDAL ENCODING ERROR] Expected a sequential embedding shape ` +
            `[1, 1, embeddingDim, sequenceLength]. Received: ${shape}`
        );
    }

    const embeddingDim = shape[2];
    const sequenceLength = shape[3];

    if (!Number.isInteger(embeddingDim) || embeddingDim <= 0) {
        throw new Error(`[SINUSOIDAL ENCODING ERROR] Invalid embedding dimension: ${embeddingDim}`);
    }

    if (!Number.isInteger(sequenceLength) || sequenceLength <= 0) {
        throw new Error(`[SINUSOIDAL ENCODING ERROR] Invalid sequence length: ${sequenceLength}`);
    }

    // These are derived from the previous layer's output shape and are
    // stored on the layer object for feedforward().
    layer_data.embeddingDim = embeddingDim;
    layer_data.maxSequenceLength = sequenceLength;

    return {
        updatedSize: size,
        updatedShape: shape,
        weights: [],
        biases: [],
        weightGrads: [],
        biasGrads: [],
        inputShape: shape,
        outputShape: shape,
        paramShape: [],
    };
};

const feedforward = (input, current_layer) => {
    const embeddingDim = current_layer.embeddingDim;
    const sequenceLength = current_layer.maxSequenceLength;

    if (input.length !== embeddingDim * sequenceLength) {
        throw new Error(`[SINUSOIDAL ENCODING ERROR] Input size (${input.length}) does not ` + `match embeddingDim * sequenceLength (${embeddingDim * sequenceLength}).`);
    }

    const output = sinusoidalPE(input, embeddingDim, sequenceLength);

    if (output.some(v => Number.isNaN(v))) {
        throw new Error("Error - output array has NaNs on Sinusoidal Encoding layer (feedforward)");
    }

    return {
        outputs: output,
        z_values: output,
        incrementor_value: 0
    };
};

module.exports = {
    initParams,
    feedforward
};

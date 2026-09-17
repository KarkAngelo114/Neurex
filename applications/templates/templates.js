const Layers = require("../../layers");

const layer = new Layers();

// Example Outputs:
// getOptimalNumHeads(768)  -> 12 (headDim = 64)
// getOptimalNumHeads(4096) -> 64 (headDim = 64)
// getOptimalNumHeads(512)  -> 8  (headDim = 64)

exports.simpleNeuralNetwork = () => {
    return [
        layer.connectedLayer(5),
        layer.connectedLayer(5),
        layer.connectedLayer(5),
    ];
}

exports.simpleCNN = (isHeadless = false) => {

    const layers = [
        layer.convolutionalLayer(8, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),
        layer.convolutionalLayer(12, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),
    ];

    let startingLayerSize = 128;
    if (!isHeadless) {
        // if not headless, adds the 3 connected layer
        for (let i = 0; i < 3; i++) {
            layers.push(layer.connectedLayer(startingLayerSize));
            startingLayerSize /= 2;
        }
    }

    return layers;
}


exports.VGG16 = () => {
    return [
        layer.convolutionalLayer(64, 1, [3, 3], 'relu', 'same'),
        layer.convolutionalLayer(64, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.convolutionalLayer(128, 1, [3, 3], 'relu', 'same'),
        layer.convolutionalLayer(128, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.convolutionalLayer(256, 1, [3, 3], 'relu', 'same'),
        layer.convolutionalLayer(256, 1, [3, 3], 'relu', 'same'),
        layer.convolutionalLayer(256, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.convolutionalLayer(512, 1, [3, 3], 'relu', 'same'),
        layer.convolutionalLayer(512, 1, [3, 3], 'relu', 'same'),
        layer.convolutionalLayer(512, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.convolutionalLayer(512, 1, [3, 3], 'relu', 'same'),
        layer.convolutionalLayer(512, 1, [3, 3], 'relu', 'same'),
        layer.convolutionalLayer(512, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.connectedLayer(4096),
        layer.connectedLayer(4096),
    ];
}

exports.LiteNet = () => {
    return [
        layer.convolutionalLayer(8, 1, [3, 3], 'relu', 'same', false),
        layer.layerNorm(),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.convolutionalLayer(16, 1, [3, 3], 'relu', 'same', false),
        layer.layerNorm(),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.convolutionalLayer(32, 1, [3, 3], 'relu', 'same', false),
        layer.layerNorm(),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.connectedLayer(128, 'relu', false),
        layer.layerNorm(),
    ];
}

exports.vanillaRNN = (units_per_cell = 3, activation_function = "tanh") => {
    return [
        layer.recurrentCell(units_per_cell, activation_function, true),
        layer.recurrentCell(units_per_cell, activation_function, true),
        layer.recurrentCell(units_per_cell, activation_function),
    ];
}


exports.GPT = (embedDim, seqLen, numHeads) => {
    if (
        !embedDim ||
        embedDim <= 0 ||
        !seqLen ||
        seqLen <= 0 ||
        !numHeads ||
        numHeads <= 0
    ) {
        throw new Error(`[ERROR embedding dimension, sequence length and numHeads must not be 0, a negative integer, null or undefined`);
    }

    return [
        layer.residualStart(),
        layer.multiHeadAttention(numHeads, true, false), // N heads, using causal masking, no biases
        layer.residualEnd(),
        layer.layerNorm(),

        layer.residualStart(),
        layer.connectedLayer(embedDim*seqLen*4, 'relu', false),
        layer.connectedLayer(embedDim*seqLen, 'linear', false),
        layer.residualEnd(),
        layer.layerNorm(),
    ];
}
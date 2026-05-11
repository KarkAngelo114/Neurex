const Layers = require("../../layers");

const layer = new Layers();

exports.simpleNeuralNetwork = () => {
    return [
        layer.connectedLayer('relu', 5),
        layer.connectedLayer('relu', 5),
        layer.connectedLayer('relu', 5),
    ];
}

exports.simpleCNN = () => {
    return [
        layer.convolutionalLayer(8, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),
        layer.convolutionalLayer(12, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),
        layer.connectedLayer('relu', 128),
        layer.connectedLayer('relu', 64),
        layer.connectedLayer('relu', 32),
    ];
}
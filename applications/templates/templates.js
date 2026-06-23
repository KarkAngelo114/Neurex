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

        layer.connectedLayer('relu', 4096),
        layer.connectedLayer('relu', 4096),
    ];
}

exports.LiteNet = () => {
    return [
        layer.convolutionalLayer(8, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.convolutionalLayer(16, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.convolutionalLayer(32, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),

        layer.connectedLayer('relu', 128),

    ];
}

exports.AutoEncoder = () => {
    return [
        layer.connectedLayer('relu', 224),
        layer.connectedLayer('relu', 112),
        layer.connectedLayer('relu', 56),
        layer.connectedLayer('relu', 28),
        layer.connectedLayer('relu', 14),
        layer.connectedLayer('relu', 7),
        layer.connectedLayer('relu', 7),
        layer.connectedLayer('relu', 7),
        layer.connectedLayer('relu', 14),
        layer.connectedLayer('relu', 28),
        layer.connectedLayer('relu', 56),
        layer.connectedLayer('relu', 112),
        layer.connectedLayer('relu', 224),
    ];
}
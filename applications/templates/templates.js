const Layers = require("../../layers");

const layer = new Layers();

exports.simpleNeuralNetwork = () => {
    return [
        layer.connectedLayer(5),
        layer.connectedLayer(5),
        layer.connectedLayer(5),
    ];
}

exports.simpleCNN = () => {
    return [
        layer.convolutionalLayer(8, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),
        layer.convolutionalLayer(12, 1, [3, 3], 'relu', 'same'),
        layer.maxPooling([2, 2], 2, 'valid'),
        layer.connectedLayer(128),
        layer.connectedLayer(64),
        layer.connectedLayer(32),
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

        layer.connectedLayer(4096),
        layer.connectedLayer(4096),
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

        layer.connectedLayer(128),

    ];
}

exports.VanillaRNN = (units_per_cell = 3, activation_function = "tanh") => {
    return [
        layer.recurrentCell(units_per_cell, activation_function, true),
        layer.recurrentCell(units_per_cell, activation_function, true),
        layer.recurrentCell(units_per_cell, activation_function, true),
    ];
}

exports.AutoEncoder = () => {
    return [
        layer.connectedLayer(224),
        layer.connectedLayer(112),
        layer.connectedLayer(64),
        layer.connectedLayer(32),
        layer.connectedLayer(16),
        layer.connectedLayer(8),
        layer.connectedLayer(4),
        layer.connectedLayer(2),
        layer.connectedLayer(2),
        layer.connectedLayer(4,'tanh'),
        layer.connectedLayer(8,'tanh'),
        layer.connectedLayer(16,'tanh'),
        layer.connectedLayer(32,'tanh'),
        layer.connectedLayer(64,'tanh'),
        layer.connectedLayer(112,'tanh'),
        layer.connectedLayer(224,'tanh'),
    ];
}
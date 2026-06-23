// loss functions used for training, updating weights and biases
// tells how "wrong" the network is in it's predictions vs actuals during training

const loss = require('../core/bindings/entry');

const mse = (predictions, actuals) => loss.mse(predictions, actuals);

const mae = (predictions, actuals) => loss.mae(predictions, actuals);

const categorical_cross_entropy = (predictions, actuals, epsilon = 1e-15) => loss.categorical_cross_entropy(predictions, actuals, epsilon);

const sparse_categorical_cross_entropy = (predictions, actuals, epsilon = 1e-15) => loss.sparse_categorical_cross_entropy(predictions, actuals, epsilon);

const binary_cross_entropy = (predictions, actuals, epsilon = 1e-15) => loss.binary_cross_entropy(predictions, actuals, epsilon);

module.exports = {
    mse,
    mae,
    categorical_cross_entropy,
    sparse_categorical_cross_entropy,
    binary_cross_entropy
};
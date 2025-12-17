let path = require('path');


let addon;

try {
    addon = require(path.join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
}
catch (error) {
    console.error(error);
}


/**
 * @function MatMul
 * @param {Array<Number>} inputs - 1D array of input features
 * @param {Array<Array<Number>>} weights - 2D array of weights
 * @param {Array<Number>} biases - 1D array of biases
 * @returns 1D array of output
 */
const MatMul = (inputs, weights, biases) => addon.MatMul(inputs, weights, biases);

/**
 * @function DeltaMatMul
 * @param {Array<Array<Number>>} weights - 2D array of weights
 * @param {Array<Number>} deltas - 1D array of output deltas from the previous layer
 * @returns 1D array of output deltas of the current layer to be use to the next layer during backpropagation
 */
const DeltaMatMul = (weights, deltas) => addon.DeltaMatMul(weights, deltas);

/**
 * @function relu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu)
 */
const relu = (input) => addon.Relu(input);

/**
 * @function sigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid)
 */
const sigmoid = (input) => addon.Sigmoid(input);

/**
 * @function tanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh)
 */
const tanh = (input) => addon.Tanh(input);

/**
 * @function softmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax)
 */
const softmax = (input) => addon.Softmax(input);

/**
 * @function linear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear)
 */
const linear = (input) => addon.Linear(input);

/**
 * @function drelu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu Derivative)
 */
const drelu = (input) => addon.DReLu(input);

/**
 * @function dsigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid Derivative)
 */
const dsigmoid = (input) => addon.DSigmoid(input);

/**
 * @function dtanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh Derivative)
 */
const dtanh = (input) => addon.DTanh(input);

/**
 * @function dsoftmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax Derivative)
 */
const dsoftmax = (input) => addon.DSoftmax(input);

/**
 * @function dlinear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear Derivative)
 */
const dlinear = (input) => addon.DLinear(input);

/**
 * 
 * @param {Number} filters 
 * @param {Number} strides 
 * @param {Array<Array<Array<Array<Number>>>>>} input 
 * @param {Array<Array<Array<Number>>>} kernels 
 * @param {Array<Number>} biases 
 * @param {String} padding 
 * @returns array of 4D feature maps
 */
const Convolve = (strides = 1,input, kernels, biases, padding = 'same') => addon.Convolve(strides, input, kernels, biases, padding);

/**
 * 
 * @param {Array<Array<Array<Array<Number>>>>} featureMaps 
 * @returns 1 stack of feature map with increasing depth
 */
const StackFeatureMaps = (featureMaps) => addon.StackFeatureMaps(featureMaps);



module.exports = {
    MatMul,
    DeltaMatMul,
    relu,
    sigmoid,
    tanh,
    softmax,
    linear,
    Convolve,
    StackFeatureMaps,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
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
 * @param {Number} OutputHeight
 * @param {Number} OutputWidth
 * @returns array of 4D feature maps
 */
const Convolve = (strides = 1,input, kernels, biases, OutputHeight, OutputWidth) => addon.Convolve(strides, input, kernels, biases, OutputHeight, OutputWidth);

/**
 * 
 * @param {Array<Array<Array<Array<Number>>>>} featureMaps 
 * @returns 1 stack of feature map with increasing depth
 */
const StackFeatureMaps = (featureMaps) => addon.StackFeatureMaps(featureMaps);

/**
 * 
 * @param {*} delta_feature_maps 
 * @param {*} kernels 
 * @param {*} padding 
 * @param {*} strides 
 * @returns 
 */
const ConvolveDelta = (delta_feature_maps, kernels, padding, strides) => addon.ConvolveDelta(delta_feature_maps, kernels, padding, strides);

/**
 * 
 * @param {*} activated_outputs 
 * @param {*} delta 
 * @param {*} layer_name 
 * @param {*} weightGrads
 * @param {*} layer_data
 * @returns 
 */
const computeWeightGradients = (activated_outputs, delta, layer_name, weightGrads, layer_data, allDeltas, layer_index) => {
    if (layer_name === "convolutional2D") {
    
        const output = addon.ComputeGradientForKernels(activated_outputs, delta, weightGrads, layer_data.strides);
        return output
    
        // return output = ComputeGradientForKernels(activated_outputs, delta, weightGrads);
    }
    else if (layer_name === "connected_layer") {
        let input = activated_outputs;
        if (Array.isArray(activated_outputs[0] || Array.isArray(activated_outputs[0][0]))) {
            input = activated_outputs.flat(Infinity);
        }
        //console.log(allDeltas[layer_index-1])
        const output = addon.ComputeGradientForDenseWeights(input, delta, weightGrads);
        return output
    }

    else {
        return weightGrads;
    }
}

// need to write a native binding for this
const ComputeGradientForKernels = (
        activated_outputs, // the activated outputs per convolutional layer assume are padded if the padding is "same") or not (the padding is "valid")
        delta, // delta output per convolutional layer
        weightGrads, // weight or (kernels)
        stride = 1 // default strides unless a value is passed
    ) => {

    const filters = weightGrads.length;
    const channels = weightGrads[0].length;
    const kernel_height = weightGrads[0][0].length;
    const kernel_width = weightGrads[0][0][0].length;

    // Loop over each filter
    let output_grad = weightGrads;
    for (let f = 0; f < filters; f++) {
        for (let c = 0; c < channels; c++) {
            for (let kh = 0; kh < kernel_height; kh++) {
                for (let kw = 0; kw < kernel_width; kw++) {
                    let grad = 0;
                    // Slide kernel over activated_outputs
                    for (let i = 0; i <= activated_outputs.length - kernel_height; i += stride) {
                        for (let j = 0; j <= activated_outputs[0].length - kernel_width; j += stride) {
                            // Multiply input patch by delta
                            grad += activated_outputs[i + kh][j + kw][c] * delta[i][j][f];
                        }
                    }
                    output_grad[f][c][kh][kw] += grad;
                }
            }
        }
    }

    return output_grad;
};


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
    ConvolveDelta,
    computeWeightGradients,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
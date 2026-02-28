/**

 These are collection of functions from the precompiled binary addon. The function that has "✅" means it uses the function from the addon.

 */

let path = require('path');
const { red, reset } = require('../../prettify');

let addon;

try {
    addon = require(path.join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
}
catch (error) {
    console.error(error);
}


/**
 * "✅"
 * @function MatMul
 * @param {Array<Number>} inputs - 1D array of input features
 * @param {Array<Array<Number>>} weights - 2D array of weights
 * @param {Array<Number>} biases - 1D array of biases
 * @returns 1D array of output
 */
const MatMul = (inputs, weights, biases) => addon.MatMul(inputs, weights, biases);

/**
 * "✅"
 * @function DeltaMatMul
 * @param {Array<Array<Number>>} weights - 2D array of weights
 * @param {Array<Number>} deltas - 1D array of output deltas from the previous layer
 * @returns 1D array of output deltas of the current layer to be use to the next layer during backpropagation
 */
const DeltaMatMul = (weights, deltas) => addon.DeltaMatMul(weights, deltas);

/**
 * "✅"
 * @function relu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu)
 */
const relu = (input) => addon.Relu(input);

/**
 * "✅"
 * @function sigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid)
 */
const sigmoid = (input) => addon.Sigmoid(input);

/**
 * "✅"
 * @function tanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh)
 */
const tanh = (input) => addon.Tanh(input);

/**
 * "✅"
 * @function softmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax)
 */
const softmax = (input) => addon.Softmax(input);

/**
 * "✅"
 * @function linear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear)
 */
const linear = (input) => addon.Linear(input);

/**
 * "✅"
 * @function drelu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu Derivative)
 */
const drelu = (input) => addon.DReLu(input);

/**
 * "✅"
 * @function dsigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid Derivative)
 */
const dsigmoid = (input) => addon.DSigmoid(input);

/**
 * "✅"
 * @function dtanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh Derivative)
 */
const dtanh = (input) => addon.DTanh(input);

/**
 * "✅"
 * @function dsoftmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax Derivative)
 */
const dsoftmax = (input) => addon.DSoftmax(input);

/**
 * "✅"
 * @function dlinear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear Derivative)
 */
const dlinear = (input) => addon.DLinear(input);

/**
 * 
 * "✅"
 * @param {Number} filters 
 * @param {Number} strides 
 * @param {Array<Array<Array<Array<Number>>>>>} input 
 * @param {Array<Array<Array<Number>>>} kernels 
 * @param {Array<Number>} biases 
 * @param {Number} OutputHeight
 * @param {Number} OutputWidth
 * @returns array of 4D feature maps
 */
const Convolve = (strides = 1,input, kernels, biases, OutputHeight, OutputWidth) => addon.Convolve(strides, input, kernels, biases, OutputHeight, OutputWidth)

/**
 * 
 * "✅"
 * @param {Array<Array<Array<Array<Number>>>>} featureMaps 
 * @returns 1 stack of feature map with increasing depth
 */
const StackFeatureMaps = (featureMaps) => addon.StackFeatureMaps(featureMaps);

/**
 * "✅"
 * Performs backpropagation convolution to find the delta of the previous layer.
 * @param {Array} padded_dilated_delta - The delta from the next layer, already dilated and padded.
 * @param {Array} kernels - The kernels/weights of the current layer [Filters][Channels][Height][Width]
 * @param {Number} inputH - The height of the input from the forward pass we are trying to reach.
 * @param {Number} inputW - The width of the input from the forward pass we are trying to reach.
 * @returns {Array} 3D Tensor [inputH][inputW][Channels]
 */
const ConvolveDelta = (padded_dilated_delta, kernels, inputH, inputW) => addon.ConvolveDelta(padded_dilated_delta, kernels, inputH, inputW);


/**
 * 
 * "✅"
 * @param {Array<Number>} params - flattened array of parameters 
 * @param {Array<Number>} grads - flattened array of grads 
 * @param {Number} learning_rate - learning rate value
 * @returns 
 */
const ApplySGD = (params, grads, learning_rate ) => addon.SGD(params, grads, learning_rate);

/**
 * 
 * "✅"
 * @param {Array<Number>} params - flattened array of parameters 
 * @param {Array<Number>} grads - flattened array of grads  
 * @param {Array<Number>} m - first momentum of average gradients vector
 * @param {Array<Number>} v - second momentum of squared average gradients vector
 * @param {Number} t - Time step counter 
 * @param {Number} learning_rate - learning rate value 
 * @param {Number} beta1 - beta1 value
 * @param {Number} beta2 - beta2 value
 * @param {Number} epsilon - epsilon constant value
 * @returns 
 */
const ApplyAdam = (params, grads, m, v, t, learning_rate, beta1, beta2, epsilon) => addon.Adam(params, grads, m, v, t, learning_rate, beta1, beta2, epsilon);

/**
 * 
 * "✅"
 * @param {Array<Number>} grad - initiated bias grads 
 * @param {Number} actual_batch_size - actual batch size 
 * @returns 
 */
const scaleGradientsForBiases = (grad, actual_batch_size) => addon.scaleGradientsForBiases(grad, actual_batch_size);

/**
 * 
 * "✅"
 * @param {*} activated_outputs 
 * @param {*} delta 
 * @param {*} layer_name 
 * @param {*} weightGrads
 * @param {*} layer_data
 * @returns 
 */
const computeWeightGradients = (activated_outputs, delta, layer_name, weightGrads, layer_data, allDeltas, layer_index) => {

    if (layer_name === "convolutionalLayer") {
    
        return ComputeGradientForKernels(activated_outputs, delta, weightGrads);
    }
    else if (layer_name === "connected_layer") {
        let input = activated_outputs;
        if (Array.isArray(activated_outputs[0] || Array.isArray(activated_outputs[0][0]))) {
            input = activated_outputs.flat(Infinity);
        }

        return addon.ComputeGradientForDenseWeights(input, delta, weightGrads);

    }

    else {
        return weightGrads;
    }
}

const ComputeGradientForKernels = (input, delta, weightGrads) => {

    const H = delta.length;
    const W = delta[0].length;
    const Cout = weightGrads.length;

    const Kh = weightGrads[0].length;
    const Kw = weightGrads[0][0].length;
    const Cin = weightGrads[0][0][0].length;

    const padH = Math.floor(Kh / 2);
    const padW = Math.floor(Kw / 2);

    for (let f = 0; f < Cout; f++) {
        for (let kh = 0; kh < Kh; kh++) {
            for (let kw = 0; kw < Kw; kw++) {
                for (let c = 0; c < Cin; c++) {

                    let sum = 0;

                    for (let h = 0; h < H; h++) {
                        for (let w = 0; w < W; w++) {

                            const inH = h + kh - padH;
                            const inW = w + kw - padW;

                            if (inH >= 0 && inH < input.length &&inW >= 0 && inW < input[0].length) {
                                sum += input[inH][inW][c] * delta[h][w][f];
                            }
                        }
                    }

                    weightGrads[f][kh][kw][c] += sum;
                }
            }
        }
    }

    return weightGrads;
};


/**
 * 
 * "✅"
 * @param {*} weightGrads 
 * @param {*} actualBatchSize 
 * @param {*} layer_name 
 * @returns 
 */
const scaleGradientsForWeights = (weightGrads, actualBatchSize, layer_name) => {
    
    if (layer_name === "connected_layer") {

        return addon.scaleGradientWeightsForConnectedLayer(weightGrads, actualBatchSize);

    }
    
    if (layer_name === "convolutionalLayer") {
        return addon.scaleGradientWeightsForConv(weightGrads, actualBatchSize);
    }

    else {
        return weightGrads;
    }
}

/**
 * 
 * "✅"
 * @param {*} biasGrads 
 * @param {*} delta 
 * @param {*} layer_name 
 * @returns 
 */
const computeBiasGradients = (biasGrads, delta, layer_name) => {

    if (layer_name === "connected_layer") {
        return addon.ComputeGradientsForDenseBiases(biasGrads, delta);
    }
    else if (layer_name === "convolutionalLayer") {
        return addon.ComputeGradientsForConvBiases(biasGrads, delta);
    }
}

/**
 * 
 * @function Marix_Mul use to multiply elements inside both arrays. Requires both arrays has same length;
 * @param {Array<Number>} flat_arr_1 - a flat array input
 * @param {Array<Number>} flat_arr_2 - a flat array input
 * @returns A flat array output after multiplying input_array_1[i] to the values of input_array_2[i]
 * @throws am error will occured if both array are not equal in length
 */
const element_wise_mul = (flat_arr_1, flat_arr_2) => {

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`${red}[EROR]------- Error: Both arrays are not equal in length. ${reset}`)

    const output = Array.from({length: flat_arr_1.length}).fill(0);
    for (let i = 0; i < flat_arr_1.length; i++) {
        output[i] = flat_arr_1[i] * flat_arr_2[i];
    }

    return output;
}


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
    computeBiasGradients,
    scaleGradientsForWeights,
    scaleGradientsForBiases,
    ApplySGD,
    ApplyAdam,
    element_wise_mul,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
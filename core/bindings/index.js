/**

 These are collection of functions from the precompiled binary addon. 
 The function that has "✅" means it uses the function from the addon. Where as if the function has also a ☑️ means it uses float32array.
 Having both ✅ and ☑️ means that it uses the function from the addon and operates on float32
 

 */

let path = require('path');
const { red, reset } = require('../../color-code');
const float_32 = require('./float32Ops');

let addon;

try {
    addon = require(path.join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
}
catch (error) {
    console.error(error);
}


/**
 * "✅☑️"
 * @function MatMul
 * @param {Float32Array} inputs - 1D float32array of input features
 * @param {Float32Array} weights - 1D float32array of weights
 * @param {Float32Array} biases - 1D float32array of biases
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @returns 1D array of output
 */
const MatMul = (inputs, weights, biases, inputSize, outputSize) => addon.MatMul(inputs, weights, biases, inputSize, outputSize);

/**
 * "☑️"
 * @function DeltaMatMul
 * @param {Float32Array} deltas - Float32Array array of output deltas from the previous layer
 * @param {Float32Array} weights - Float32Array array of weights
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @returns 1D array of output deltas of the current layer to be use to the next layer during backpropagation
 */
const DeltaMatMul = (deltas, weights, inputSize, outputSize) => addon.DeltaMatMul(deltas, weights, inputSize, outputSize);

/**
 * "✅☑️"
 * @function relu
 * @param {Float32Array} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu)
 */
const relu = (input) => addon.Relu(input)

/**
 * "✅☑️"
 * @function sigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid)
 */
const sigmoid = (input) => addon.Sigmoid(input);

/**
 * "✅☑️"
 * @function tanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh)
 */
const tanh = (input) => addon.Tanh(input);

/**
 * "✅☑️"
 * @function softmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax)
 */
const softmax = (input) => addon.Softmax(input);

/**
 * "✅☑️"
 * @function linear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear)
 */
const linear = (input) => addon.Linear(input); 

/**
 * "✅☑️"
 * @function drelu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu Derivative)
 */
const drelu = (input) => addon.DReLu(input);

/**
 * "✅☑️"
 * @function dsigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid Derivative)
 */
const dsigmoid = (input) => addon.DSigmoid(input);

/**
 * "✅☑️"
 * @function dtanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh Derivative)
 */
const dtanh = (input) => addon.DTanh(input);

/**
 * "✅☑️"
 * @function dsoftmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax Derivative)
 */
const dsoftmax = (input) => addon.DSoftmax(input)

/**
 * "✅☑️"
 * @function dlinear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear Derivative)
 */
const dlinear = (input) => addon.DLinear(input);

/**
 * "☑️"
 * @param {Float32Array} input 
 * @param {Number} inputH 
 * @param {Number} inputW 
 * @param {Number} channels 
 * @param {Number} padTop 
 * @param {Number} padBottom 
 * @param {Number} padLeft 
 * @param {Number} padRight 
 * @returns padded tensor
 */
const applyPadding = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => float_32.ApplyPadding_Float32(input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight)

/**
 * "✅☑️"
 * @param {Float32Array} input - padded input
 * @param {Float32Array} kernels - kernels of the current layer
 * @param {Float32Array} biases - biases for each kernels of the current layer
 * @param {number} strides - determines how many pixels the kernel will skip
 * @param {number} outputH - expected output height
 * @param {number} outputW - expected ouptut width
 * @param {number} num_filters - number of filters
 * @param {number} kernel_height - kernel height
 * @param {number} kernel_width - kernel width
 * @param {number} depth - depth
 * @param {number} inputH - input height of the padded input
 * @param {number} inputW - input widht of the padded input
 * @returns output float32 of the convolution
 */
const Convolve = (input, kernels, biases, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW) => addon.Convolve(input, kernels, biases, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW);


/**
 * "☑️" dilate the input inserting 0s
 * @param {Float32Array} delta 
 * @param {Array<Number>} shape_array 
 * @param {Number} strides 
 * @returns Dilated delta
 */
const Dilate_Input = (delta, shape_array, strides) => float_32.DilateDelta_Float32(delta, shape_array, strides);

/**
 * "☑️"
 * @param {Float32Array} params - the kernels 
 * @param {Numnber} f - number of filters 
 * @param {Number} kh - kernel height
 * @param {Number} kw - kernel width 
 * @param {Numbwe} d - depth of the kernel
 * @returns float32array of parameters
 */
const rotate_kernels = (params, f, kh, kw, d) => float_32.RotateKernels_Float32(params, f, kh, kw, d);


/**
 * "✅☑️" Perform delta convolution
 * @param {Float32Array} input - padded delta
 * @param {Array<Number>} padded_delta_shape - padded delta shape
 * @param {Float32Array} kernels - rotated kernels 
 * @param {Array<Number>} kernel_shape - kernel shapes arrange as [f][kh][kw][c] 
 * @param {Number} strides
 * @returns output delta convolution
 */
const ConvolveDelta = (input, padded_delta_shape,  kernels, kernel_shape, oh, ow) => addon.ConvolveDelta(input, padded_delta_shape, kernels, kernel_shape, oh, ow);


/**
 * 
 * "✅☑️"
 * @param {Float32Array} params - flattened array of parameters 
 * @param {Float32Array} grads - flattened array of grads 
 * @param {Number} learning_rate - learning rate value
 * @returns 
 */
const ApplySGD = (params, grads, learning_rate ) => addon.SGD(params, grads, learning_rate);

/**
 * 
 * "✅☑️"
 * @param {Float32Array} params - flattened array of parameters 
 * @param {Float32Array} grads - flattened array of grads
 * @param {Number} learning_rate - learning rate value 
 * @param {Float32Array} m - first momentum of average gradients vector
 * @param {Float32Array} v - second momentum of squared average gradients vector
 * @param {Number} t - Time step counter 
 * @param {Number} epsilon - epsilon constant value
 * @param {Number} beta1 - beta1 value
 * @param {Number} beta2 - beta2 value
 * @returns 
 */
const ApplyAdam = (params, grads, learning_rate, m, v, t, epsilon, beta1, beta2) => addon.Adam(params, grads, m, v, t, learning_rate, beta1, beta2, epsilon);

/**
 * 
 * "✅☑️"
 * @param {Float32Array>} activated_outputs 
 * @param {Float32Array>} delta 
 * @param {Float32Array>} weightGrads
 * @param {Array<Number>} weightShape
 * @returns float32array of accumulated weight gradients
 */
const computeWeightGradientsForWeightsInConnectedLayer = (activations, delta, weightGrads, inputSize, outputSize) => addon.computeWeightGradientsForWeightsInConnectedLayer(activations, delta, weightGrads, inputSize, outputSize);

/**
 * "✅☑️"
 * @param {Float32Array} activated_outputs - activated outputs during feedforward
 * @param {Float32Array} delta - delta outputs during backpropagation 
 * @param {Float32Array} weightGrads - initiated weight gradients for accumulation 
 * @param {number} inputH - input height
 * @param {number} inputW - input width 
 * @param {number} C_in - input channels
 * @param {number} Out_H - output height
 * @param {number} Out_W - output width 
 * @param {number} C_out - output channel 
 * @param {number} kh - kernel height 
 * @param {number} kw -kernel width
 * @returns 
 */
const ComputeGradientForKernels = (activated_outputs, delta, weightGrads, inputH, inputW, C_in, Out_H, Out_W, C_out, kh, kw) => addon.computeKernelGradients(activated_outputs, delta, weightGrads, inputH, inputW, C_in, Out_H, Out_W, C_out, kh, kw);

/**
 * "✅☑️"
 * @param {Float32Array>} biasGrads 
 * @param {Float32Array>} delta 
 * @returns float32array of accumulated bias gradients
 */
const computeBiasGradsForConnected_Layer = (biasGrads, delta) => addon.computeBiasGradsForConnected_Layer(biasGrads, delta);

/**
 * "✅☑️"
 * @param {Float32Array} grads - bias grads in float32array 
 * @param {Float32Array} deltas - float32array delta 
 * @returns Accumulated bias gradients in float32array
 */
const computeBiasGradsForConv = (grads, deltas, oh, ow, num_filters) => addon.computeBiasGradsForConv(grads, deltas, oh, ow, num_filters);

/**
 * "✅☑️"
 * @param {Float32Array} grad - accumulated gradients
 * @param {Number} batchSize - batch size
 * @returns A float32 array of scaled gradients
 */
const scaleGrads = (grad, batchSize) => addon.scaleGrad(grad, batchSize)

/**
 * 
 * "✅☑️"
 * @function Marix_Mul use to multiply elements inside both arrays. Requires both arrays has same length;
 * @param {Array<Number>} flat_arr_1 - a flat array input
 * @param {Array<Number>} flat_arr_2 - a flat array input
 * @returns A flat array output after multiplying input_array_1[i] to the values of input_array_2[i]
 * @throws am error will occured if both array are not equal in length
 */
const element_wise_mul = (flat_arr_1, flat_arr_2) => {

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`${red}[ERROR]------- Error: Both arrays are not equal in length. ${reset}`);

    return addon.element_wise_mul(flat_arr_1, flat_arr_2);
}


module.exports = {
    MatMul,
    DeltaMatMul,
    relu,
    sigmoid,
    tanh,
    softmax,
    linear,
    applyPadding,
    Convolve,
    Dilate_Input,
    rotate_kernels,
    ConvolveDelta,
    computeWeightGradientsForWeightsInConnectedLayer,
    ComputeGradientForKernels,
    computeBiasGradsForConnected_Layer,
    computeBiasGradsForConv,
    scaleGrads,
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
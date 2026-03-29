/**

 These are collection of functions from the precompiled binary addon. 
 The function that has "✅" means it uses the function from the addon. Where as if the function has also a ☑️ means it uses float32array.
 Having both ✅ and ☑️ means that it uses the function from the addon and operates on float32
 

 */

let path = require('path');
const { red, reset } = require('../../prettify');
const float_32 = require('./float32Ops');

let addon;

try {
    addon = require(path.join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
}
catch (error) {
    console.error(error);
}


/**
 * "☑️"
 * @function MatMul
 * @param {Array<Number>} inputs - 1D array of input features
 * @param {Array<Array<Number>>} weights - 2D array of weights
 * @param {Array<Number>} biases - 1D array of biases
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @returns 1D array of output
 */
const MatMul = (inputs, weights, biases, inputSize, outputSize) => float_32.MatMul_Float32(inputs, weights, biases, inputSize, outputSize);

/**
 * "☑️"
 * @function DeltaMatMul
 * @param {Float32Array} deltas - Float32Array array of output deltas from the previous layer
 * @param {Float32Array} weights - Float32Array array of weights
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @returns 1D array of output deltas of the current layer to be use to the next layer during backpropagation
 */
const DeltaMatMul = (deltas, weights, inputSize, outputSize) => float_32.DeltaMatMul_Float32(deltas, weights, inputSize, outputSize);

/**
 * "☑️"
 * @function relu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu)
 */
const relu = (input) => float_32.relu_float32(input);

/**
 * "☑️"
 * @function sigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid)
 */
const sigmoid = (input) => float_32.sigmoid_float32(input);

/**
 * "☑️"
 * @function tanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh)
 */
const tanh = (input) => float_32.tanh_float32(input);

/**
 * "☑️"
 * @function softmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax)
 */
const softmax = (input) => float_32.softmax_float32(input);

/**
 * "☑️"
 * @function linear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear)
 */
const linear = (input) => float_32.linear_float32(input);

/**
 * "☑️"
 * @function drelu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu Derivative)
 */
const drelu = (input) => float_32.drelu_float32(input);

/**
 * "☑️"
 * @function dsigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid Derivative)
 */
const dsigmoid = (input) => float_32.dsigmoid_float32(input);

/**
 * "☑️"
 * @function dtanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh Derivative)
 */
const dtanh = (input) => float_32.dtanh_float32(input);

/**
 * "☑️"
 * @function dsoftmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax Derivative)
 */
const dsoftmax = (input) => float_32.dsoftmax_float32(input);

/**
 * "☑️"
 * @function dlinear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear Derivative)
 */
const dlinear = (input) => float_32.dlinear_float32(input);

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
 * "☑️"
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
const Convolve = (input, kernels, biases, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW) => float_32.Convolve_Float32(input, kernels, biases, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW);


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
 * "☑️" Perform delta convolution
 * @param {Float32Array} input - padded delta
 * @param {Array<Number>} padded_delta_shape - padded delta shape
 * @param {Float32Array} kernels - rotated kernels 
 * @param {Array<Number>} kernel_shape - kernel shapes arrange as [f][kh][kw][c] 
 * @param {Number} strides
 * @returns output delta convolution
 */
const ConvolveDelta = (input, padded_delta_shape,  kernels, kernel_shape, oh, ow) => float_32.ConvolveDelta_Float32(input, padded_delta_shape, kernels, kernel_shape, oh, ow);


/**
 * 
 * "☑️"
 * @param {Float32Array} params - flattened array of parameters 
 * @param {Float32Array} grads - flattened array of grads 
 * @param {Number} learning_rate - learning rate value
 * @returns 
 */
const ApplySGD = (params, grads, learning_rate ) => float_32.ApplySGD_float32(params, grads, learning_rate);

/**
 * 
 * "☑️"
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
const ApplyAdam = (params, grads, learning_rate, m, v, t, epsilon, beta1, beta2) => float_32.ApplyAdam_float32(params, grads, learning_rate, m, v, t, epsilon, beta1, beta2);


/**
 * 
 * "☑️"
 * @param {Float32Array>} activated_outputs 
 * @param {Float32Array>} delta 
 * @param {Float32Array>} weightGrads
 * @param {Array<Number>} weightShape
 * @returns float32array of accumulated weight gradients
 */
const computeWeightGradientsForWeightsInConnectedLayer = (activations, delta, weightGrads, inputSize, outputSize) => float_32.computeWeightGradientsForWeightsInConnectedLayer_float32(activations, delta, weightGrads, inputSize, outputSize);

/**
 * "☑️"
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
const ComputeGradientForKernels = (activated_outputs, delta, weightGrads, inputH, inputW, C_in, Out_H, Out_W, C_out, kh, kw) => float_32.computeKernelGradients_Float32(activated_outputs, delta, weightGrads, inputH, inputW, C_in, Out_H, Out_W, C_out, kh, kw);

/**
 * "☑️"
 * @param {Float32Array>} biasGrads 
 * @param {Float32Array>} delta 
 * @returns float32array of accumulated bias gradients
 */
const computeBiasGradsForConnected_Layer = (biasGrads, delta) => float_32.computeBiasGradsForConnected_Layer_float32(biasGrads, delta);

/**
 * "☑️"
 * @param {Float32Array} grads - bias grads in float32array 
 * @param {Float32Array} deltas - float32array delta 
 * @returns Accumulated bias gradients in float32array
 */
const computeBiasGradsForConv = (grads, deltas, oh, ow, num_filters) => float_32.computeBiasGradsForConv_Float32(grads, deltas, oh, ow, num_filters);

/**
 * "☑️"
 * @param {Float32Array} grad - accumulated gradients
 * @param {Number} batchSize - batch size
 * @returns A float32 array of scaled gradients
 */
const scaleGrads = (grad, batchSize) => float_32.scaleGrads_float32(grad, batchSize)

/**
 * 
 * "☑️"
 * @function Marix_Mul use to multiply elements inside both arrays. Requires both arrays has same length;
 * @param {Array<Number>} flat_arr_1 - a flat array input
 * @param {Array<Number>} flat_arr_2 - a flat array input
 * @returns A flat array output after multiplying input_array_1[i] to the values of input_array_2[i]
 * @throws am error will occured if both array are not equal in length
 */
const element_wise_mul = (flat_arr_1, flat_arr_2) => {
    const output = new Float32Array(flat_arr_1.length);

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`${red}[ERROR]------- Error: Both arrays are not equal in length. ${reset}`);

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
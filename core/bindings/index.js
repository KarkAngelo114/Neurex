/**

 These are collection of functions from the precompiled binary addon. 
 The function that has "✅" means it uses the function from the addon. Where as if the function has also a ☑️ means it uses float32array.
 Having both ✅ and ☑️ means that it uses the function from the addon and operates on float32

 */

let path = require('path');
const {BooleanAvailability} = require('../../gpu/modeSelector'); 
const { red, reset, yellow } = require('../../color-code');
const float32_Modules = require('./float32Ops');
const CPU_Based_addon = require(path.join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'neurex-core-native.node'));

let functions;


const init = () => {    

    try {

        /* 
        * This library might support GPU acceleration soon so we need proper branching of exposed functions. The default fallback are the functions from "float32Ops" module where everything is written in Javascript.
        * Ideal if on different environment and setup like:
        * 
        * - on different OS but the prebuilt binaries are not compiled to the target OS environment, so default to use "float32_Modules"
        * - on OSes where the prebuilt binaries are compatible, but no GPU available, use the "CPU_Based_addon"
        * - on OSes where the prebuilt binaries are compatible, and has GPU available, then use the GPU based addon 
        */


        const {hasGPU, force_Use_Default_JS_Float32_Module, data} = BooleanAvailability();

        if (force_Use_Default_JS_Float32_Module) {
            console.log(`${yellow}\n[INFO]${reset} Using Javascript-float32 modules`);
            functions = float32_Modules;
            return;
        }

        if (hasGPU) {
            console.log(`\n⚡ I, ${path.join(__dirname,"..", "..", "gpu", "gpu_init.js")} found a device ${yellow}${data.devices[0].gpu}${reset} whose vendor is ${yellow}${data.devices[0].vendor}${reset} with a memory of ${yellow}${(Number(data.devices[0].globalMemBytes) / (1024 ** 3 )).toFixed(2)} GB${reset}.`);
            console.log(`⚡ Initializing GPU accelerated training....`);
            // use the CPU-based library for now
            functions = CPU_Based_addon;
            return
        }

        if (!hasGPU) {
            console.log(`${yellow}\n[INFO]${reset} Neurex will use the default CPU-based compiled binary.`);
            functions = CPU_Based_addon;
            return;
        }

        
    }
    catch (error) {
        console.error(error);
    }
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
const MatMul = (inputs, weights, biases, inputSize, outputSize) => functions.MatMul(inputs, weights, biases, inputSize, outputSize);
/**
 * "☑️"
 * @function DeltaMatMul
 * @param {Float32Array} deltas - Float32Array array of output deltas from the previous layer
 * @param {Float32Array} weights - Float32Array array of weights
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @returns 1D array of output deltas of the current layer to be use to the next layer during backpropagation
 */
const DeltaMatMul = (deltas, weights, inputSize, outputSize) => functions.DeltaMatMul(deltas, weights, inputSize, outputSize);

/**
 * "✅☑️"
 * @function relu
 * @param {Float32Array} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu)
 */
const relu = (input) => functions.Relu(input)

/**
 * "✅☑️"
 * @function sigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid)
 */
const sigmoid = (input) => functions.Sigmoid(input);

/**
 * "✅☑️"
 * @function tanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh)
 */
const tanh = (input) => functions.Tanh(input);

/**
 * "✅☑️"
 * @function softmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax)
 */
const softmax = (input) => functions.Softmax(input);

/**
 * "✅☑️"
 * @function linear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear)
 */
const linear = (input) => functions.Linear(input); 

/**
 * "✅☑️"
 * @function drelu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu Derivative)
 */
const drelu = (input) => functions.DReLu(input);

/**
 * "✅☑️"
 * @function dsigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid Derivative)
 */
const dsigmoid = (input) => functions.DSigmoid(input);

/**
 * "✅☑️"
 * @function dtanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh Derivative)
 */
const dtanh = (input) => functions.DTanh(input);

/**
 * "✅☑️"
 * @function dsoftmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax Derivative)
 */
const dsoftmax = (input) => functions.DSoftmax(input)

/**
 * "✅☑️"
 * @function dlinear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear Derivative)
 */
const dlinear = (input) => functions.DLinear(input);

/**
 * "✅☑️"
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
const applyPadding = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => functions.ApplyPadding(input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight)

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
const Convolve = (input, kernels, biases, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW) => functions.Convolve(input, kernels, biases, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW);


/**
 * "✅☑️" dilate the input inserting 0s
 * @param {Float32Array} delta 
 * @param {Array<Number>} shape_array 
 * @param {Number} strides 
 * @returns Dilated delta
 */
const Dilate_Input = (delta, shape_array, strides) => functions.DilateDelta(delta, shape_array, strides);

/**
 * "✅☑️"
 * @param {Float32Array} params - the kernels 
 * @param {Numnber} f - number of filters 
 * @param {Number} kh - kernel height
 * @param {Number} kw - kernel width 
 * @param {Numbwe} d - depth of the kernel
 * @returns float32array of parameters
 */
const rotate_kernels = (params, f, kh, kw, d) => functions.RotateKernels(params, f, kh, kw, d);


/**
 * "✅☑️" Perform delta convolution
 * @param {Float32Array} input - padded delta
 * @param {Array<Number>} padded_delta_shape - padded delta shape
 * @param {Float32Array} kernels - rotated kernels 
 * @param {Array<Number>} kernel_shape - kernel shapes arrange as [f][kh][kw][c] 
 * @param {Number} strides
 * @returns output delta convolution
 */
const ConvolveDelta = (input, padded_delta_shape,  kernels, kernel_shape, oh, ow) => functions.ConvolveDelta(input, padded_delta_shape, kernels, kernel_shape, oh, ow);


/**
 * 
 * "✅☑️"
 * @param {Float32Array} params - flattened array of parameters 
 * @param {Float32Array} grads - flattened array of grads 
 * @param {Number} learning_rate - learning rate value
 * @returns 
 */
const ApplySGD = (params, grads, learning_rate ) => functions.SGD(params, grads, learning_rate);

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
const ApplyAdam = (params, grads, learning_rate, m, v, t, epsilon, beta1, beta2) => functions.Adam(params, grads, m, v, t, learning_rate, beta1, beta2, epsilon);

/**
 * 
 * "✅☑️"
 * @param {Float32Array>} activated_outputs 
 * @param {Float32Array>} delta 
 * @param {Float32Array>} weightGrads
 * @param {Array<Number>} weightShape
 * @returns float32array of accumulated weight gradients
 */
const computeWeightGradientsForWeightsInConnectedLayer = (activations, delta, weightGrads, inputSize, outputSize) => functions.computeWeightGradientsForWeightsInConnectedLayer(activations, delta, weightGrads, inputSize, outputSize);

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
const ComputeGradientForKernels = (activated_outputs, delta, weightGrads, inputH, inputW, C_in, Out_H, Out_W, C_out, kh, kw) => functions.computeKernelGradients(activated_outputs, delta, weightGrads, inputH, inputW, C_in, Out_H, Out_W, C_out, kh, kw);

/**
 * "✅☑️"
 * @param {Float32Array>} biasGrads 
 * @param {Float32Array>} delta 
 * @returns float32array of accumulated bias gradients
 */
const computeBiasGradsForConnected_Layer = (biasGrads, delta) => functions.computeBiasGradsForConnected_Layer(biasGrads, delta);

/**
 * "✅☑️"
 * @param {Float32Array} grads - bias grads in float32array 
 * @param {Float32Array} deltas - float32array delta 
 * @returns Accumulated bias gradients in float32array
 */
const computeBiasGradsForConv = (grads, deltas, oh, ow, num_filters) => functions.computeBiasGradsForConv(grads, deltas, oh, ow, num_filters);

/**
 * "✅☑️"
 * @param {Float32Array} grad - accumulated gradients
 * @param {Number} batchSize - batch size
 * @returns A float32 array of scaled gradients
 */
const scaleGrads = (grad, batchSize) => functions.scaleGrad(grad, batchSize)

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

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`${red}[ERROR]------- Error: Both arrays are not equal in length. array1: ${flat_arr_1.length} | array2:${flat_arr_2.length} ${reset}`);
    
    return functions.element_wise_mul(flat_arr_1, flat_arr_2);
}

/**
 * "✅☑️"
 * @function MaxPool
 * @param {Float32Array} input - current input passed down to this layer 
 * @param {Array<Number>} poolSize - pool size of the sliding window
 * @param {Array<Number>} inputShape - input shape of the current tensor
 * @param {Array<Number>} outputShape - output shape of the tensor
 * @param {Number} strides - determines how many pixels it will skipped
 */
const MaxPool = (input, poolSize, inputShape, outputShape, strides) => functions.MaxPooling(input, poolSize, inputShape, outputShape, strides);

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
    MaxPool,
    init,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
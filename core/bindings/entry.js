/**

 These are collection of functions from the precompiled binary addon. 
 The function that has "✅" means it uses the function from the addon. Where as if the function has also a ☑️ means it uses float32array.
 Having both ✅ and ☑️ means that it uses the function from the addon and operates on float32

 */

let path = require('path');
const {BooleanAvailability} = require('../../gpu/modeSelector'); 
const { red, reset, yellow } = require('../../color-code');
const float32_Modules = require('./float32Ops');
const { getGlobalParams } = require('../../gpu/globals');
let addon;

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

        addon = require(path.join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'neurex-core-native.node'));

        if (hasGPU) {
            console.log(`\n⚡ I, ${path.join(__dirname,"..", "..", "gpu", "gpu_init.js")} found a device ${yellow}${data.devices[0].gpu}${reset} whose vendor is ${yellow}${data.devices[0].vendor}${reset} with a memory of ${yellow}${(Number(data.devices[0].globalMemBytes) / (1024 ** 3 )).toFixed(2)} GB${reset}.`);
            const kernelSource = path.join(__dirname, "..", "..", "gpu", "kernels");
            
            console.log("Compiling kernels...");
            const res = addon.Init_GPU(kernelSource);

            if (!res.ok) {
                console.warn(`\n${yellow}[WARN]${reset} GPU Kernel initialization failed. Failing back to CPU`);
                console.log(res.error);
                // if "failed", we need to set the global boolean state on C++ to false to use CPU-based functions.
                addon.setOnGPU(false);
                functions = addon;
                return;
            }
            
            console.log("Kernels successfully compiled...");
            addon.setOnGPU(true);
            functions = addon;
            return
        }

        if (!hasGPU && !force_Use_Default_JS_Float32_Module) {
            console.log(`${yellow}\n[INFO]${reset} Neurex will use the optimized CPU functions`);
            addon.setOnGPU(false);
            functions = addon;
            return;
        }

        
    }
    catch (error) {
        console.error(error);
    }
}


/**
 *  "✅☑️"
 * @function getEmbeddings
 * @param {Array<Number>} tokenVector an array of token vector 
 * @param {Number} embeddingDim embedding dim value
 * @param {Number} pointer pointer value corresponding to the global parameter of weights and biases 
 * @param {Number} outputTemplatePointer pointer value correspondind to the output template tensor 
 * @returns {Float32Array} flattened embeddings
 */
const getEmbeddings = (tokenVector, embeddingDim, pointer, outputTemplatePointer) => functions.getEmbeddings(Array.from(tokenVector), embeddingDim, getGlobalParams().globalWeights[pointer], outputTemplatePointer);

/**
 * "✅☑️"
 * @param {Array<Number>} activated_outputs activation outputs. During feedfoward, the activation outputs before going to the embedding layer is actually the raw token array
 * @param {Float32Array} delta float32array delta 
 * @param {Float32Array} weightGrads initialized 0s
 * @param {Number} dim - Embedding Dim
 * @returns {Float32Array} 
 */
const returnEmbeddings = (activated_outputs, delta, weightGrads, dim) => functions.returnEmbeddings(Array.from(activated_outputs), delta, weightGrads, dim);

/**
 * "✅☑️"
 * @function MatMul
 * @param {Float32Array} inputs - 1D float32array of input features
 * @param {Float32Array} weights - 1D float32array of weights
 * @param {Float32Array} biases - 1D float32array of biases
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @param {Number} pointer - a pointer that will be use to index the corresponding parameter from global params
 * @returns 1D array of output
 */
const MatMul = (inputs, inputSize, outputSize, pointer, outputTemplatePointer) => functions.MatMul(
    inputs, 
    inputSize, 
    outputSize, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer], 
    outputTemplatePointer
);

/**
 * "✅☑️"
 * @function DeltaMatMul
 * @param {Float32Array} deltas - Float32Array array of output deltas from the previous layer
 * @param {Float32Array} weights - Float32Array array of weights
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @param {Number} pointer - a pointer that will be use to index the corresponding parameter from global params
 * @returns 1D array of output deltas of the current layer to be use to the next layer during backpropagation
 */
const DeltaMatMul = (deltas, inputSize, outputSize, pointer) => functions.DeltaMatMul(
    deltas, 
    inputSize, 
    outputSize, 
    getGlobalParams().globalWeights[pointer]
);

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
 * @param {Float32Array} p predictions array 
 * @param {Float32Array} a actuals array 
 * @returns loss output
 */
const mse = (p, a) => functions.mse(new Float32Array(p), new Float32Array(a));

/**
 * "✅☑️"
 * @param {Float32Array} p predictions array 
 * @param {Float32Array} a actuals array 
 * @returns loss output
 */
const mae = (p, a) => functions.mae(new Float32Array(p), new Float32Array(a));

/**
 * "✅☑️"
 * @param {Float32Array} p predictions array 
 * @param {Float32Array} a actuals array 
 * @param {Number} epsilon epsilon value. Default is `1e-15`
 * @returns loss output
 */
const categorical_cross_entropy = (p, a, epsilon = 1e-15) => float32_Modules.categorical_cross_entropy(new Float32Array(p), new Float32Array(a), epsilon);

/**
 * "✅☑️"
 * @param {Float32Array} p predictions array 
 * @param {Float32Array} a actuals array 
 * @param {Number} epsilon epsilon value. Default is `1e-15`
 * @returns loss output
 */
const sparse_categorical_cross_entropy = (p, a, epsilon = 1e-15) => float32_Modules.sparse_categorical_cross_entropy(new Float32Array(p), a, epsilon);

/**
 * "✅☑️"
 * @param {Float32Array} p predictions array 
 * @param {Float32Array} a actuals array 
 * @param {Number} epsilon epsilon value. Default is `1e-15`
 * @returns loss output
 */
const binary_cross_entropy = (p, a, epsilon = 1e-15) => float32_Modules.binary_cross_entropy(new Float32Array(p), new Float32Array(a), epsilon);

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
const applyPadding = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => functions.ApplyPadding(input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight);

/**
 * "✅☑️"
 * @param {Float32Array} input input to perform convolution
 * @param {Number} strides stride value
 * @param {Array<Number>} outputShape [oH, oW]
 * @param {Array<Number>} kernelShape [num_filters, Kh, Kw, channels]
 * @param {Array<Number>} inputShape [iH, iW] 
 * @param {Number} pointer pointer value to fetch corresponding parameters of the layer from the global store
 * @param {Number} outputTemplatePointer pointer value to fetch allocated tensor of the layer from the global store
 * @returns {Float32Array} convolution result
 */
const Convolve = (input, strides, outputShape, kernelShape, inputShape, pointer, outputTemplatePointer) => functions.Convolve(
    input, 
    strides, 
    outputShape, 
    kernelShape, 
    inputShape, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer], 
    outputTemplatePointer
);

/**
 * "✅☑️" dilate the input inserting 0s
 * @param {Float32Array} input 
 * @param {Array<Number>} shape_array 
 * @param {Number} strides 
 * @returns {Object} {data, dilatedHeight, dilatedWidth}
 */
const Dilate_Input = (input, shape_array, strides) => functions.DilateInput(input, shape_array, strides);

/**
 * "✅☑️"
 * @param {Float32Array} input input tensors
 * @param {Array<Number>} deltaShape delta shape: [Hp, Wp, C_in]
 * @param {Array<Number>} kernel_shape kernel shape: [F, KH, KW, C_k]
 * @param {Array<Number>} outputShape output shape: [oH, oW]
 * @param {Numer} pointer pointer value to fetch parameters from the global store
 * @param {Nunber} stride stride value
 * @returns {Float32Array} convolve result
 */
const ConvolveDelta = (input, deltaShape, kernel_shape, outputShape, pointer, stride = 1) => functions.ConvolveDelta(
    input, 
    deltaShape, 
    kernel_shape, 
    outputShape, 
    getGlobalParams().globalWeights[pointer],
    stride
);

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
 * @param {Float32Array} input inputs that is already activated by and activation function
 * @param {Float32Array} delta delta input
 * @param {Float32Array} ZeroedGrads zero gradients for accumulation
 * @param {Array<Number>} inputShape input shape: [inputH, inputW, Cin]
 * @param {Array<Number>} outputShape output shape: [H, W, Cout]
 * @param {Array<Number>} kernelSize kernel size: [Kh, Kw]
 * @param {Number} stride stride value. Default value is `1`
 * @returns accumulated gradients
 */
const ComputeGradientForKernels = (input, delta, ZeroedGrads, inputShape, outputShape, kernelSize, stride = 1) => functions.computeKernelGradients(input, delta, ZeroedGrads, inputShape, outputShape, kernelSize, stride);

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
 * @function 
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
 * 
 * "✅☑️"
 * @function
 * @param {Array<Number>} flat_arr_1 - a flat array input
 * @param {Array<Number>} flat_arr_2 - a flat array input
 * @returns A flat array output after subtracting input_array_1[i] to the values of input_array_2[i]
 * @throws am error will occured if both array are not equal in length
 */
const element_wise_sub = (flat_arr_1, flat_arr_2) => {

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`${red}[ERROR]------- Error: Both arrays are not equal in length. array1: ${flat_arr_1.length} | array2:${flat_arr_2.length} ${reset}`);
    return functions.element_wise_sub(new Float32Array(flat_arr_1), new Float32Array(flat_arr_2));
}

/**
 * "✅☑️"
 * @param {Foat32Array} arr1 a flat array input
 * @param {Foat32Array} arr2 a flat array input
 * @param {Foat32Array} arr3 a flat array input
 * @returns a flat array after performing `(arr1[i] - arr2[i]) * arr3[i]`
 * @throws {Error} - if any of the input array are not equal in length
 */
const scaleDiff = (arr1, arr2, arr3) => {
    if (arr1.length !== arr2.length || arr2.length !== arr3.length || arr1.length !== arr3.length) {
        throw new Error(`${red}[ERROR]------- Error: All arrays must be equal in length. array1: ${arr1.length} | array2: ${arr2.length} | array3: ${arr3.length} ${reset}`);
    }

    return functions.scaleDiff(new Float32Array(arr1), new Float32Array(arr2), new Float32Array(arr3));
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
const MaxPool = (input, poolSize, inputShape, outputShape, strides, outputTemplatePointer) => functions.MaxPooling(input, poolSize, inputShape, outputShape, strides, outputTemplatePointer);

/**
 * "✅☑️"
 * @param {Float32Array} delta incoming 
 * @param {Int32Array} indices an array containing the index corresponding to the max pooled value
 * @param {Number} h height of the input tensor
 * @param {*} w width of the input tensor
 * @param {*} d depth of the input tensor
 * @returns 
 */
const MaxPoolDelta = (delta, indices, h, w, d) => functions.MaxPoolDelta(delta, indices, h, w, d);



module.exports = {
    getEmbeddings,
    returnEmbeddings,
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
    ConvolveDelta,
    computeWeightGradientsForWeightsInConnectedLayer,
    ComputeGradientForKernels,
    computeBiasGradsForConnected_Layer,
    computeBiasGradsForConv,
    scaleGrads,
    ApplySGD,
    ApplyAdam,
    element_wise_mul,
    element_wise_sub,
    MaxPool,
    MaxPoolDelta,
    init,
    scaleDiff,
    mse,
    mae,
    categorical_cross_entropy,
    sparse_categorical_cross_entropy,
    binary_cross_entropy,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
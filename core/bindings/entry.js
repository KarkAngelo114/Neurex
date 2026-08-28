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
const { transpose2D, unpackQKVO } = require('../../utils/utils');
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
 * @returns {Float32Array} flattened embeddings
 */
const getEmbeddings = (tokenVector, embeddingDim, pointer) => functions.getEmbeddings(Array.from(tokenVector), embeddingDim, getGlobalParams().globalWeights[pointer]);

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
const MatMul = (inputs, inputSize, outputSize, pointer) => functions.MatMul(
    inputs, 
    inputSize, 
    outputSize, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer], 
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
 * @param {Float32Array} arr1 - Float32Array input
 * @param {Float32Array} arr2 - Float32Array input
 * @returns - 1D array of activated features (Using Softmax Derivative)
 */
const dsoftmax = (arr1, arr2) => functions.DSoftmax(arr1, arr2);

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
 * @returns {Float32Array} convolution result
 */
const Convolve = (input, strides, outputShape, kernelShape, inputShape, pointer) => functions.Convolve(
    input, 
    strides, 
    outputShape, 
    kernelShape, 
    inputShape, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer],
);

/**
 * "✅☑️" dilate the input inserting 0s
 * @param {Float32Array} input 
 * @param {Array<Number>} shape_array 
 * @param {Number} strides 
 * @returns {{data: Float32Array, dilatedHeight: Number, dilatedWidth: Number}} {data, dilatedHeight, dilatedWidth}
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
 * @param {Float32Array} velocity - array of calculated velocity
 * @param {Number} learning_rate - learning rate value
 * @param {Number} momentum - momentum value
 * @returns {{params: Float32Array, velocity: Float32Array}}
 */
const ApplySGD = (params, grads, velocity, lr, momentum = 0.9) => functions.SGD(params, grads, velocity, lr, momentum = 0.9);

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
 * "✅☑️" performs X[i] /= scaling_value
 * @param {Float32Array} input - float32Array input
 * @param {Number} scalingValue - scaling value
 * @returns A float32 array of scaled outpur
 */
const scale = (input, scalingValue) => functions.scale(input, scalingValue);

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
const MaxPool = (input, poolSize, inputShape, outputShape, strides) => functions.MaxPooling(input, poolSize, inputShape, outputShape, strides);

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

/**
 * "✅☑️"
 * @param {Float32Array} input input vector
 * @param {Float32Array} prevHiddenState hidden temporal state
 * @param {Array<Number>} inputWeightShape input weight shape
 * @param {Array<Number>} recurrentWeightShape recurrent weight shape
 * @param {Number} pointer value to reference the weights and biases 
 * @returns 
 */
const recurrentMatMul = (input, prevHiddenState, inputWeightShape, recurrentWeightShape, pointer) => functions.recurrentMatMul(
    input, 
    prevHiddenState,
    inputWeightShape, 
    recurrentWeightShape, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer],
);


/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Array<Number>} inputWeightShape 
 * @param {Array<Number>} recurrentWeightShape 
 * @param {Number} pointer 
 * @returns 
 */
const recurrentTimeDelta = (input, inputWeightShape, recurrentWeightShape, pointer) => functions.recurrentTimeDelta(
    input, 
    inputWeightShape,
    recurrentWeightShape,
    getGlobalParams().globalWeights[pointer], 
);

/**
 * "✅☑️"
 * @param {Float32Array} activation_outputs 
 * @param {Float32Array} deltas 
 * @param {Array<Float32Array>} hiddenStates 
 * @param {Array<Float32Array>} deltaTs 
 * @param {Float32Array} weightGrads 
 * @param {Array<Number>} weightShape 
 * @param {Number} sequenceLength 
 * @returns 
 */
const recurrentWeightGradsAccumulation = (activation_outputs, deltas, hiddenStates, deltaTs, weightGrads, weightShape, sequenceLength) => functions.recurrentWeightGradsAccumulation(
    activation_outputs, 
    deltas, 
    hiddenStates, 
    deltaTs, 
    weightGrads, 
    weightShape, 
    sequenceLength
);

/**
 * "✅☑️"
 * @param {Float32Array} biasGrads 
 * @param {Array<Float32Array>} deltaTs 
 * @param {Number} sequenceLength 
 * @param {Number} units 
 * @returns 
 */
const recurrentBiasGradsAccumulation = (biasGrads, deltaTs, sequenceLength, units) => functions.recurrentBiasGradsAccumulation(
    biasGrads, 
    deltaTs, 
    sequenceLength, 
    units
);

/**
 * "✅☑️"
 * @param {Float32Array} grads 
 * @param {Number} threshold 
 * @returns {Float32Array}
 */
const gradientClipping = (grads, threshold) => functions.gradientClipping(grads, threshold);

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Array<Number>} inputShape 
 * @param {Array<Number>} outputShape 
 * @param {Number} strides 
 * @param {Number} filters 
 * @param {Array<Number>} weightShape 
 * @param {Number} pointer 
 * @returns {Float32Array} trans conv output.
 */
const transConv = (input, inputShape, outputShape, strides, filters, weightShape, pointer) => functions.transConv(
    input, 
    inputShape, 
    outputShape, 
    strides, 
    filters, 
    weightShape, 
    getGlobalParams().globalWeights[pointer], 
    getGlobalParams().globalBiases[pointer],
);

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Array<Number>} inputShape 
 * @param {Array<Number>} outputShape 
 * @param {Number} strides 
 * @param {Number} filters 
 * @param {Array<Number>} weightShape 
 * @param {pointer} pointer 
 * @returns {Float32Array} delta tensor to be projected
 */
const transConvBackward = (input, inputShape, outputShape, strides, filters, weightShape, pointer) => functions.transConvBackward(
    input,
    inputShape,
    outputShape,
    strides,
    filters,
    weightShape,
    getGlobalParams().globalWeights[pointer], 
);

/**
 * "✅☑️"
 * @param {*} activation_outputs 
 * @param {*} delta 
 * @param {*} zeroGradAccumulator 
 * @param {*} strides 
 * @param {*} filters 
 * @param {*} inputShape 
 * @param {*} outputShape 
 * @param {*} weightShape 
 * @returns 
 */
const accumulateKernelGradsForTransConv = (activation_outputs, delta, zeroGradAccumulator, strides, filters, inputShape, outputShape, weightShape) => functions.accumulateKernelGradsForTransConv(
    activation_outputs,
    delta, 
    zeroGradAccumulator,
    strides,
    filters, 
    inputShape, 
    outputShape, 
    weightShape
);

/**
 * "✅☑️"
 * @param {Float32Array} arr1 
 * @param {Float32Array} arr2 
 * @param {Number} inputSize 
 * @param {Number} outputSize 
 * @returns {Float32Array}
 */
const dotProduct = (arr1, arr2, inputSize, outputSize) => functions.dotProduct(arr1, arr2, inputSize, outputSize);

/**
 * 
 * @param {Float32Array} input 
 * @param {Object} layerData 
 * @param {Number} pointer 
 * @returns 
 */
const CoreAttention = (input, layerData, pointer) => {
    const { embedDim, dkRoot, seqLen } = layerData;
    const weights = getGlobalParams().globalWeights[pointer];
    const biases = getGlobalParams().globalBiases[pointer];
    
    const {Q_weights, Q_bias, K_weights, K_bias, V_weights, V_bias} = unpackQKVO(weights, biases, null, null, embedDim);

    const Q = new Float32Array(seqLen * embedDim);
    const K = new Float32Array(seqLen * embedDim);
    const V = new Float32Array(seqLen * embedDim);

    for (let t = 0; t < seqLen; t++) {
        const tokenVec = input.subarray(t * embedDim, (t + 1) * embedDim);
        Q.set(functions.MatMul(tokenVec, embedDim, embedDim, Q_weights, Q_bias), t * embedDim);
        K.set(functions.MatMul(tokenVec, embedDim, embedDim, K_weights, K_bias), t * embedDim);
        V.set(functions.MatMul(tokenVec, embedDim, embedDim, V_weights, V_bias), t * embedDim);
    }

    const transpose_K = transpose2D(K, seqLen, embedDim);

    const scores = new Float32Array(seqLen * seqLen);
    for (let t = 0; t < seqLen; t++) {
        const Qrow = Q.subarray(t * embedDim, (t + 1) * embedDim);
        const rowScores = dotProduct(Qrow, transpose_K, embedDim, seqLen); // inputSize=embedDim, outputSize=seqLen
        scores.set(rowScores, t * seqLen);
    }

    const scaledvals = scale(scores, dkRoot);

    const softmaxOutput = new Float32Array(seqLen * seqLen);
    for (let t = 0; t < seqLen; t++) {
        const row = scaledvals.subarray(t * seqLen, (t + 1) * seqLen);
        softmaxOutput.set(softmax(row), t * seqLen);
    }

    const output = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const srow = softmaxOutput.subarray(t * seqLen, (t + 1) * seqLen);
        const orow = dotProduct(srow, V, seqLen, embedDim); // inputSize=seqLen, outputSize=embedDim
        output.set(orow, t * embedDim);
    }

    layerData.cache = {
        X: input,
        Q: Q, 
        K: K, 
        V: V,
        S: softmaxOutput
    };
    
    return output;
}

/**
 * 
 * @param {Float32Array} incomingDelta 
 * @param {Object} layerData 
 * @param {Number} pointer 
 */
const CoreAttentionBackward = (incomingDelta, layerData, pointer) => {
    const { embedDim, dkRoot, seqLen } = layerData;
    const { Q, K, V, S: storedS } = layerData.cache; 
    const weights = getGlobalParams().globalWeights[pointer];

    // just like in feedforward, we unpack the weights, but we pass "null" to the 2nd - 4th argument of the function because we only want the weights for QKV
    const {Q_weights, K_weights, V_weights} = unpackQKVO(weights, null, null, null, embedDim);

    const transpose_V = transpose2D(V, seqLen, embedDim); 
    const dS = new Float32Array(seqLen * seqLen);
    for (let t = 0; t < seqLen; t++) {
        const deltaRow = incomingDelta.subarray(t * embedDim, (t + 1) * embedDim);
        dS.set(dotProduct(deltaRow, transpose_V, embedDim, seqLen), t * seqLen);
    }

    const transpose_S = transpose2D(storedS, seqLen, seqLen); // Sᵀ
    const dV = new Float32Array(seqLen * embedDim);
    for (let k = 0; k < seqLen; k++) {
        const sCol = transpose_S.subarray(k * seqLen, (k + 1) * seqLen);
        dV.set(dotProduct(sCol, incomingDelta, seqLen, embedDim), k * embedDim);
    }

    // apply the softmax derivative (Jacobian matrix)
    const dScaled = new Float32Array(seqLen * seqLen);
    for (let t = 0; t < seqLen; t++) {
        const sRow = storedS.subarray(t * seqLen, (t + 1) * seqLen);
        const dSRow = dS.subarray(t * seqLen, (t + 1) * seqLen);
        
        // we pass the sRow (storedS during feedfoward) and dSRow
        dScaled.set(functions.DSoftmax(sRow, dSRow), t * seqLen);
    }

    const dScores = scale(dScaled, dkRoot);

    const dQ = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const dScoreRow = dScores.subarray(t * seqLen, (t + 1) * seqLen);
        dQ.set(dotProduct(dScoreRow, K, seqLen, embedDim), t * embedDim);
    }

    const transpose_dScores = transpose2D(dScores, seqLen, seqLen);
    const dK = new Float32Array(seqLen * embedDim);
    for (let k = 0; k < seqLen; k++) {
        const col = transpose_dScores.subarray(k * seqLen, (k + 1) * seqLen);
        dK.set(dotProduct(col, Q, seqLen, embedDim), k * embedDim);
    }

    const transpose_Qw = transpose2D(Q_weights, embedDim, embedDim);
    const transpose_Kw = transpose2D(K_weights, embedDim, embedDim);
    const transpose_Vw = transpose2D(V_weights, embedDim, embedDim);

    const dX = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const dQrow = dQ.subarray(t * embedDim, (t + 1) * embedDim);
        const dKrow = dK.subarray(t * embedDim, (t + 1) * embedDim);
        const dVrow = dV.subarray(t * embedDim, (t + 1) * embedDim);

        const fromQ = dotProduct(dQrow, transpose_Qw, embedDim, embedDim);
        const fromK = dotProduct(dKrow, transpose_Kw, embedDim, embedDim);
        const fromV = dotProduct(dVrow, transpose_Vw, embedDim, embedDim);

        for (let d = 0; d < embedDim; d++) {
            dX[t * embedDim + d] = fromQ[d] + fromK[d] + fromV[d];
        }
    }


    layerData.cache = {
        dQ: dQ,
        dK: dK,
        dV: dV
    }

    return dX;
}

const CoreMultiHeadAttention = (input, layerData, pointer) => {

    const {embedDim, seqLen, numHeads, headDim, dkRoot} = layerData;

    const weights = getGlobalParams().globalWeights[pointer];
    const biases = getGlobalParams().globalBiases[pointer];

    // 1. Unpack Q, K, V, and O
    const { Q_weights, Q_bias, K_weights, K_bias, V_weights, V_bias, O_weights, O_bias } = unpackQKVO(weights, biases, null, null, embedDim, true);

    // 2. Project Input to Q, K, V [seqLen, embedDim]
    const Q = new Float32Array(seqLen * embedDim);
    const K = new Float32Array(seqLen * embedDim);
    const V = new Float32Array(seqLen * embedDim);

    for (let t = 0; t < seqLen; t++) {
        const tokenVec = input.subarray(t * embedDim, (t + 1) * embedDim);
        Q.set(functions.MatMul(tokenVec, embedDim, embedDim, Q_weights, Q_bias), t * embedDim);
        K.set(functions.MatMul(tokenVec, embedDim, embedDim, K_weights, K_bias), t * embedDim);
        V.set(functions.MatMul(tokenVec, embedDim, embedDim, V_weights, V_bias), t * embedDim);
    }

    const mhaOutput = new Float32Array(seqLen * embedDim);
    const S_per_head = [];

    // 3. Process Each Head Independently
    for (let h = 0; h < numHeads; h++) {
        const headOffset = h * headDim;

        // Extract Head-specific Q, K, V slices [seqLen, headDim]
        const Q_h = new Float32Array(seqLen * headDim);
        const K_h = new Float32Array(seqLen * headDim);
        const V_h = new Float32Array(seqLen * headDim);

        for (let t = 0; t < seqLen; t++) {
            Q_h.set(Q.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            K_h.set(K.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            V_h.set(V.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
        }

        // just like in simple attention, we ran dot product each Q * (K^T)
        const transpose_K_h = transpose2D(K_h, seqLen, headDim);
        const scores = new Float32Array(seqLen * seqLen);
        for (let t = 0; t < seqLen; t++) {
            const Qrow = Q_h.subarray(t * headDim, (t + 1) * headDim);
            const rowScores = dotProduct(Qrow, transpose_K_h, headDim, seqLen);
            scores.set(rowScores, t * seqLen);
        }

        const scaledVals = scale(scores, dkRoot);
        const softmaxOutput = new Float32Array(seqLen * seqLen);

        for (let t = 0; t < seqLen; t++) {
            const row = scaledVals.subarray(t * seqLen, (t + 1) * seqLen);
            const softmaxRow = softmax(row);
            softmaxOutput.set(softmaxRow, t * seqLen);
        }
        S_per_head.push(softmaxOutput);

        // Multiply Softmax Scores with V_h & Write Back to Concatenated Array
        for (let t = 0; t < seqLen; t++) {
            const srow = softmaxOutput.subarray(t * seqLen, (t + 1) * seqLen);
            const headOutRow = dotProduct(srow, V_h, seqLen, headDim);
            
            // Insert back into the target head position in mhaOutput
            const targetIdx = t * embedDim + headOffset;
            mhaOutput.set(headOutRow, targetIdx);
        }
    }

    // 4. Final Linear Projection (W_O) [seqLen, embedDim]
    const finalOutput = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const mhaRow = mhaOutput.subarray(t * embedDim, (t + 1) * embedDim);
        finalOutput.set(functions.MatMul(mhaRow, embedDim, embedDim, O_weights, O_bias), t * embedDim);
    }

    layerData.cache = {
        X: input,
        Q, K, V,
        mhaOutput,
        S_perHead: S_per_head
    };

    return finalOutput;
};

const CoreMultiHeadAttentionBackward = (incomingDelta, layerData, pointer) => {
    const { embedDim, seqLen, numHeads, headDim, dkRoot, cache } = layerData;
    const { Q, K, V, S_perHead } = cache;

    const weights = getGlobalParams().globalWeights[pointer];
    const {Q_weights, K_weights, V_weights, O_weights} = unpackQKVO(weights, null, null, null, embedDim, true);

    // first we get the dMHAoutput by projecting the incoming delta to transposed O_weights
    const transposed_O = transpose2D(O_weights, embedDim, embedDim);
    const dMhaOutput = new Float32Array(embedDim * seqLen);
    for (let i = 0; i < seqLen; i++) {
        const incomingDeltaRow = incomingDelta.subarray(i * embedDim, (i + 1) * embedDim);
        dMhaOutput.set(dotProduct(incomingDeltaRow, transposed_O, embedDim, embedDim), i * embedDim);
    }

    // Per head
    const dQ = new Float32Array(seqLen * embedDim);
    const dK = new Float32Array(seqLen * embedDim);
    const dV = new Float32Array(seqLen * embedDim);

    for (let h = 0; h < numHeads; h++) {
        const headOffset = h * headDim;
        const S = S_perHead[h];

        // slice this head's Q_h, K_h, V_h, and its share of dMhaOutput
        const Q_h = new Float32Array(seqLen * headDim);
        const K_h = new Float32Array(seqLen * headDim);
        const V_h = new Float32Array(seqLen * headDim);
        const dHeadOut = new Float32Array(seqLen * headDim);
        for (let t = 0; t < seqLen; t++) {
            Q_h.set(Q.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            K_h.set(K.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            V_h.set(V.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            dHeadOut.set(dMhaOutput.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
        }

        const transpose_Vh = transpose2D(V_h, seqLen, headDim);
        const dS = new Float32Array(seqLen * seqLen);
        for (let t = 0; t < seqLen; t++) {
            dS.set(dotProduct(dHeadOut.subarray(t * headDim, (t + 1) * headDim), transpose_Vh, headDim, seqLen), t * seqLen);
        }

        const transpose_S = transpose2D(S, seqLen, seqLen);
        const dV_h = new Float32Array(seqLen * headDim);
        for (let k = 0; k < seqLen; k++) {
            dV_h.set(dotProduct(transpose_S.subarray(k * seqLen, (k + 1) * seqLen), dHeadOut, seqLen, headDim), k * headDim);
        }

        const dScaled = new Float32Array(seqLen * seqLen);
        for (let t = 0; t < seqLen; t++) {
            const sRow = S.subarray(t * seqLen, (t + 1) * seqLen);
            const dSRow = dS.subarray(t * seqLen, (t + 1) * seqLen);
            dScaled.set(dsoftmax(sRow, dSRow), t * seqLen);
        }
        const dScores = scale(dScaled, dkRoot);

        const dQ_h = new Float32Array(seqLen * headDim);
        for (let t = 0; t < seqLen; t++) {
            dQ_h.set(dotProduct(dScores.subarray(t * seqLen, (t + 1) * seqLen), K_h, seqLen, headDim), t * headDim);
        }
        const transpose_dScores = transpose2D(dScores, seqLen, seqLen);
        const dK_h = new Float32Array(seqLen * headDim);
        for (let k = 0; k < seqLen; k++) {
            dK_h.set(dotProduct(transpose_dScores.subarray(k * seqLen, (k + 1) * seqLen), Q_h, seqLen, headDim), k * headDim);
        }

        // write this head's contribution back into the FULL embedDim-wide buffers
        for (let t = 0; t < seqLen; t++) {
            dQ.set(dQ_h.subarray(t * headDim, (t + 1) * headDim), t * embedDim + headOffset);
            dK.set(dK_h.subarray(t * headDim, (t + 1) * headDim), t * embedDim + headOffset);
            dV.set(dV_h.subarray(t * headDim, (t + 1) * headDim), t * embedDim + headOffset);
        }
    }

    layerData.cache.dQ = dQ;
    layerData.cache.dK = dK;
    layerData.cache.dV = dV;
    layerData.cache.dMhaOutput = dMhaOutput;

    const transpose_Qw = transpose2D(Q_weights, embedDim, embedDim);
    const transpose_Kw = transpose2D(K_weights, embedDim, embedDim);
    const transpose_Vw = transpose2D(V_weights, embedDim, embedDim);
    const dX = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const fromQ = dotProduct(dQ.subarray(t * embedDim, (t + 1) * embedDim), transpose_Qw, embedDim, embedDim);
        const fromK = dotProduct(dK.subarray(t * embedDim, (t + 1) * embedDim), transpose_Kw, embedDim, embedDim);
        const fromV = dotProduct(dV.subarray(t * embedDim, (t + 1) * embedDim), transpose_Vw, embedDim, embedDim);
        for (let d = 0; d < embedDim; d++) {
            dX[t * embedDim + d] = fromQ[d] + fromK[d] + fromV[d]
        };
    }

    return dX;
}


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
    transConv,
    transConvBackward,
    Dilate_Input,
    ConvolveDelta,
    computeWeightGradientsForWeightsInConnectedLayer,
    ComputeGradientForKernels,
    accumulateKernelGradsForTransConv,
    computeBiasGradsForConnected_Layer,
    computeBiasGradsForConv,
    scale,
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
    recurrentMatMul,
    recurrentTimeDelta,
    recurrentWeightGradsAccumulation,
    recurrentBiasGradsAccumulation,
    gradientClipping,
    CoreAttention,
    dotProduct,
    CoreAttentionBackward,
    CoreMultiHeadAttention,
    CoreMultiHeadAttentionBackward,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
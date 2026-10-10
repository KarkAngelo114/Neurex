/**

 These are collection of functions from the precompiled binary addon. 
 The function that has "✅" means it uses the function from the addon. Where as if the function has also a ☑️ means it uses float32array in JS.
 Having both ✅ and ☑️ means that it uses the function from the addon and operates on float32

 */

let path = require('path');
const float32_Modules = require('./float32Ops');
const { globalState } = require('../../gpu/modeSelector'); 
const { red, reset, yellow } = require('../../color-code');
const { getGlobalParams } = require('../../gpu/globals');

let addon;
let functions;

const init = () => {    

    try {

        const {computeBackend, device} = globalState();

        if (computeBackend === "pure-js") {
            console.log(`${yellow}[INFO]${reset} Defaulting to pure JS implementation. To speed things up, consider using native C++ bindigs by setting your compute backend to "cpu" if you have dedicated GPUs, consider using "opencl"`);
            functions = float32_Modules;
            return;
        }

        addon = require(path.join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'neurex-core-native.node'));

        if (computeBackend === "cpu") {
            console.log(`${yellow}[INFO]${reset} Neurex will use native binaries optimized for CPU-based functions`);
            addon.setComputeBackendType(computeBackend);
            functions = addon;
            return;
        }

        if (computeBackend === "opencl" && device) {
            const vramGB = (Number(device.globalMemBytes) / (1024 ** 3)).toFixed(2);

            console.log(
`\n⚡ I, ${path.join(__dirname,"..", "..", "gpu", "gpu_init.js")} found a device:`+ 
`\n- GPU: ${yellow}${device.gpu}${reset}` +
`\n- Vendor: (${yellow}${device.vendor}${reset})` +
`\n- VRAM capacity: ${yellow}${vramGB} GB${reset}` +
`\n- Compute units: ${yellow}${device.computeUnits}${reset} compute units` +
`\n- OpenCL version: ${device.openclVersion.trim()}`
            );

            const kernelSource = path.join(__dirname, "..", "..", "gpu", "kernels");

            console.log("Compiling kernels...");
            const res = addon.Init_GPU(kernelSource, device.index); // to compile OpenCL kernels

            if (!res.ok) {
                console.warn(`\n${yellow}[WARN]${reset} GPU kernel initialization failed. Falling back to CPU.`);
                console.warn(res.error);
                addon.setComputeBackendType("cpu"); // force to cpu if kernel compilation failed even OpenCL successfully detect GPU
                functions = addon;
                return;
            }

            console.log(`${yellow}[INFO]${reset} Kernels successfully compiled on ${device.gpu}.`);
            addon.setComputeBackendType(computeBackend); // pass device type on C++ addon if "opencl"
            functions = addon;
            return;
        }

        if (computeBackend === "cuda") {
            // coming soon..
        }

        
    }
    catch (error) {
        console.error(error);
    }
}

const shutdown = (modelID) => {
    if (BooleanAvailability().hasGPU) {
        addon.shutdown();
        addon.ReleaseParams(modelID);
    }
}


/**
 *  "✅☑️"
 * @function getEmbeddings
 * @param {Array<Number>} tokenVector an array of token vector 
 * @param {Number} embeddingDim embedding dim value
 * @param {Number} pointer pointer value corresponding to the global parameter of weights and biases
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns {Float32Array} flattened embeddings
 */
const getEmbeddings = (tokenVector, embeddingDim, pointer, modelID, layerID) => functions.getEmbeddings(
    Array.from(tokenVector), 
    embeddingDim, 
    getGlobalParams(modelID).globalWeights[pointer], 
    pointer,
    modelID,
    layerID
)

const sinusoidalPE = (input, embeddingDim, sequenceLength, modelID, layerID) => functions.SinusoidalPositionalEncoding(
    input, 
    embeddingDim, 
    sequenceLength,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Array<Number>} activated_outputs activation outputs. During feedfoward, the activation outputs before going to the embedding layer is actually the raw token array
 * @param {Float32Array} delta float32array delta 
 * @param {Float32Array} weightGrads initialized 0s
 * @param {Number} dim - Embedding Dim
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns {Float32Array} 
 */
const returnEmbeddings = (activated_outputs, delta, weightGrads, dim, modelID, layerID) => functions.returnEmbeddings(
    Array.from(activated_outputs), 
    delta, 
    weightGrads, 
    dim,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @function MatMul
 * @param {Float32Array} inputs - 1D float32array of input features
 * @param {Float32Array} weights - 1D float32array of weights
 * @param {Float32Array} biases - 1D float32array of biases
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @param {Number} pointer - a pointer that will be use to index the corresponding parameter from global params
 * @param {String} modelID model ID
 * @param {String} layerID 
 */
const MatMul = (inputs, inputSize, outputSize, pointer, modelID, layerID) => functions.MatMul(
    inputs, 
    inputSize, 
    outputSize, 
    getGlobalParams(modelID).globalWeights[pointer], 
    getGlobalParams(modelID).globalBiases[pointer], 
    pointer,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @function DeltaMatMul
 * @param {Float32Array} deltas - Float32Array array of output deltas from the previous layer
 * @param {Float32Array} weights - Float32Array array of weights
 * @param {Number} inputSize - the output size of the previous layer is the input size of this layer
 * @param {Number} outputSize - the layer size of this layer
 * @param {Number} pointer - a pointer that will be use to index the corresponding parameter from global params
 * @param {String} modelID model ID
 * @param {String} layerID
 */
const DeltaMatMul = (deltas, inputSize, outputSize, pointer, modelID, layerID) => functions.DeltaMatMul(
    deltas, 
    inputSize, 
    outputSize, 
    getGlobalParams(modelID).globalWeights[pointer],
    pointer,
    modelID,
    layerID
);

/**
 * 
 * @param {Float32Array} activations 
 * @param {Float32Array} deltas 
 * @param {Float32Array} weightGrads 
 * @param {Float32Array} biasGrads 
 * @param {Array<Number>} weightShape
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns {{ weightGrads: Float32Array, biasGrads: Float32Array }}
 */
const accumulateWeightsAndBiasGradsForConnectedLayer  = (activations, deltas, weightGrads, biasGrads, weightShape, modelID, layerID) => functions.accumulateWeightsAndBiasGradsForConnectedLayer(
    activations,
    deltas,
    weightGrads,
    biasGrads,
    weightShape,
    modelID,
    layerID
);


/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const relu = (input, modelID, layerID) => functions.Relu(input, modelID, layerID);
/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const sigmoid = (input, modelID, layerID) => functions.Sigmoid(input, modelID, layerID);
/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {String} modelID
 * @param {String} layerID 
 * @returns 
 */
const tanh = (input, modelID, layerID) => functions.Tanh(input, modelID, layerID);
/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const softmax = (input, modelID, layerID) => functions.Softmax(input, modelID, layerID);
/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const linear = (input, modelID, layerID) => functions.Linear(input, modelID, layerID); 
/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Float32Array} _ - reserved float32Array argument space
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const drelu = (input, _, modelID, layerID) => functions.DReLu(input, modelID, layerID);
/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Float32Array} _ - reserved float32Array argument space
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const dsigmoid = (input, _, modelID, layerID) => functions.DSigmoid(input, modelID, layerID);
/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Float32Array} _ - reserved float32Array argument space
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const dtanh = (input, _, modelID, layerID) => functions.DTanh(input, modelID, layerID);

/**
 *  "✅☑️"
 * @param {Float32Array} arr1 
 * @param {Float32Array} arr2 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const dsoftmax = (arr1, arr2, modelID, layerID) => functions.DSoftmax(arr1, arr2, modelID, layerID);
/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Float32Array} _ - reserved float32Array argument space
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const dlinear = (input, _, modelID, layerID) => functions.DLinear(input, modelID, layerID);
/**
 * "✅☑️"
 * @param {Float32Array} p 
 * @param {Float32Array} a 
 * @returns 
 */
const mse = (p, a) => functions.mse(new Float32Array(p), new Float32Array(a));
/**
 * "✅☑️"
 * @param {Float32Array} p 
 * @param {Float32Array} a 
 * @returns 
 */
const mae = (p, a) => functions.mae(new Float32Array(p), new Float32Array(a));
/**
 * "✅☑️"
 * @param {Float32Array} p 
 * @param {Arrayy<Number>} a 
 * @param {Number} epsilon 
 * @returns 
 */
const categorical_cross_entropy = (p, a, epsilon = 1e-15) => float32_Modules.categorical_cross_entropy(new Float32Array(p), new Float32Array(a), epsilon);
/**
 * "✅☑️"
 * @param {Float32Array} p 
 * @param {Arrayy<Number>} a 
 * @param {Number} epsilon 
 * @returns 
 */
const sparse_categorical_cross_entropy = (p, a, epsilon = 1e-15) => float32_Modules.sparse_categorical_cross_entropy(new Float32Array(p), a, epsilon);
/**
 * "✅☑️"
 * @param {Float32Array} p 
 * @param {Arrayy<Number>} a 
 * @param {Number} epsilon 
 * @returns 
 */
const binary_cross_entropy = (p, a, epsilon = 1e-15) => float32_Modules.binary_cross_entropy(new Float32Array(p), new Float32Array(a), epsilon);

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Array<Number>} inputShape 
 * @param {Array<Number>} outputShape 
 * @param {Array<Number>} kernelShape 
 * @param {Number} pointer 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const ConvolveForward = (input, inputShape, outputShape, kernelShape, pointer, modelID, layerID) => functions.ConvolveForward(
    input,
    inputShape,
    outputShape,
    kernelShape,
    getGlobalParams(modelID).globalWeights[pointer], 
    getGlobalParams(modelID).globalBiases[pointer],
    pointer,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Array<Number>} OutputProjectionShape 
 * @param {Array<Number>} deltaInputShape 
 * @param {Array<Number>} kernelShape 
 * @param {Number} pointer 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const ConvolveBackward = (input, OutputProjectionShape, deltaInputShape, kernelShape, pointer, modelID, layerID) => functions.ConvolveBackward(
    input,
    OutputProjectionShape,
    deltaInputShape,
    kernelShape,
    getGlobalParams(modelID).globalWeights[pointer],
    pointer,
    modelID,
    layerID
);

/**
 * 
 * @param {Float32Array} activations 
 * @param {Float32Array} deltas 
 * @param {Float32Array} weightGrads 
 * @param {Float32Array} biasGrads 
 * @param {Array<Number>} inputShape 
 * @param {Array<Number>} outputShape 
 * @param {Array<Number>} kernelShape 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns {{weightGrads: Float32Array, biasGrads: Float32Array}}
 */
const AccumulateWeightAndBiasGradsForConv = (activations, deltas, weightGrads, biasGrads, inputShape, outputShape, kernelShape, modelID, layerID) => float32_Modules.AccumulateWeightAndBiasGradsForConv(
    activations,
    deltas,
    weightGrads,
    biasGrads,
    inputShape,
    outputShape,
    kernelShape,
    1,
    modelID,
    layerID
);


/**
 * 
 * "✅☑️"
 * @param {Float32Array} params flattened array of parameters 
 * @param {Float32Array} grads flattened array of grads 
 * @param {Float32Array} velocity array of calculated velocity
 * @param {Number} learning_rate learning rate value
 * @param {Number} momentum momentum value
 * @param {number} pointer pointer value to access corresponding parameters based on Model ID
 * @param {String} paramType specify if it's a weights or biases parameter
 * @param {String} modelID unique indentification string to use a model's corresponding structured parameters
 * @param {String} layerID unique indentification string to use a layer's corresponding structured parameters
 * @returns {{params: Float32Array, velocity: Float32Array}}
 */
const ApplySGD = (params, grads, velocity, lr, momentum = 0.9, pointer, paramType, modelID, layerID) => functions.SGD(params, grads, velocity, lr, momentum, pointer, paramType, modelID, layerID);

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
 * @param {number} pointer pointer value to access corresponding parameters based on Model ID
 * @param {String} paramType specify if it's a weights or biases parameter
 * @param {String} modelID unique indentification string to use a model's corresponding structured parameters
 * @param {String} layerID unique indentification string to use a layer's corresponding structured parameters
 * @returns 
 */
const ApplyAdam = (params, grads, learning_rate, m, v, t, epsilon, beta1, beta2, pointer, paramType, modelID, layerID) => functions.Adam(params, grads, m, v, t, learning_rate, beta1, beta2, epsilon, pointer, paramType, modelID, layerID);

/**
 * "✅☑️"
 * @param {Float32Array} params flattened array of parameters 
 * @param {Float32Array} grads flattened array of grads
 * @param {Float32Array} sqAvg flattened array of moving squared average
 * @param {Number} lr learning rate value 
 * @param {Number} epsilon epsilon value
 * @param {Number} decayRate decay rate value
 * @param {number} pointer pointer value to access corresponding parameters based on Model ID
 * @param {String} paramType specify if it's a weights or biases parameter
 * @param {String} modelID unique indentification string to use a model's corresponding structured parameters
 * @param {String} layerID unique indentification string to use a layer's corresponding structured parameters
 * @returns {{ params: Float32Array, sqAvg: Float32Array }}
 */
const ApplyRMSProp = (params, grads, sqAvg, lr, epsilon, decayRate, pointer, paramType, modelID, layerID) => functions.RMSProp(params, grads, sqAvg, lr, epsilon, decayRate, pointer, paramType, modelID, layerID);

/**
 * "✅☑️" performs X[i] /= scaling_value
 * @param {Float32Array} input - float32Array input
 * @param {Number} scalingValue - scaling value
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns A float32 array of scaled outpur
 */
const scale = (input, scalingValue, modelID, layerID) => functions.scale(input, scalingValue, modelID, layerID);

/**
 * "✅☑️"
 * @param {*} flat_arr_1 
 * @param {*} flat_arr_2 
 * @param {*} pointer 
 * @param {*} modelID 
 * @returns 
 */
const element_wise_mul = (flat_arr_1, flat_arr_2, modelID, layerID) => {

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`${red}[ERROR]------- Error: Both arrays are not equal in length. array1: ${flat_arr_1.length} | array2:${flat_arr_2.length} ${reset}`);
    
    return functions.element_wise_mul(flat_arr_1, flat_arr_2, modelID, layerID);
}

/**
 * 
 * "✅☑️"
 * @function
 * @param {Array<Number>} flat_arr_1 - a flat array input
 * @param {Array<Number>} flat_arr_2 - a flat array input
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns A flat array output after subtracting input_array_1[i] to the values of input_array_2[i]
 * @throws am error will occured if both array are not equal in length
 */
const element_wise_sub = (flat_arr_1, flat_arr_2, modelID, layerID) => {

    if (flat_arr_1.length != flat_arr_2.length) throw new Error(`${red}[ERROR]------- Error: Both arrays are not equal in length. array1: ${flat_arr_1.length} | array2:${flat_arr_2.length} ${reset}`);
    return functions.element_wise_sub(new Float32Array(flat_arr_1), new Float32Array(flat_arr_2), modelID, layerID);
}

/**
 * 
 * "✅☑️"
 * @function
 * @param {Array<Number>} arr1 - a flat array input
 * @param {Array<Number>} arr2- a flat array input
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns A flat array output after adding input_array_1[i] to the values of input_array_2[i]
 * @throws am error will occured if both array are not equal in length
 */
const element_wise_add = (arr1, arr2, modelID, layerID) => {
    if (arr1.length != arr2.length) throw new Error(`[ERROR] Error: Both arrays are not equal in length. array1: ${arr1.length} | array2:${arr2.length}`);

    return functions.element_wise_add(arr1, arr2, modelID, layerID);
}

/**
 * "✅☑️"
 * @param {Foat32Array} arr1 a flat array input
 * @param {Foat32Array} arr2 a flat array input
 * @param {Foat32Array} arr3 a flat array input
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns a flat array after performing `(arr1[i] - arr2[i]) * arr3[i]`
 * @throws {Error} - if any of the input array are not equal in length
 */
const scaleDiff = (arr1, arr2, arr3, modelID, layerID) => {
    if (arr1.length !== arr2.length || arr2.length !== arr3.length || arr1.length !== arr3.length) {
        throw new Error(`${red}[ERROR]------- Error: All arrays must be equal in length. array1: ${arr1.length} | array2: ${arr2.length} | array3: ${arr3.length} ${reset}`);
    }

    return functions.scaleDiff(new Float32Array(arr1), new Float32Array(arr2), new Float32Array(arr3), modelID, layerID);
}

/**
 * "✅☑️"
 * @function MaxPool
 * @param {Float32Array} input - current input passed down to this layer 
 * @param {Array<Number>} poolSize - pool size of the sliding window
 * @param {Array<Number>} inputShape - input shape of the current tensor
 * @param {Array<Number>} outputShape - output shape of the tensor
 * @param {Number} strides - determines how many pixels it will skipped
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 */
const MaxPool = (input, poolSize, inputShape, outputShape, strides, modelID, layerID) => functions.MaxPooling(
    input, 
    poolSize, 
    inputShape, 
    outputShape, 
    strides,
    modelID, 
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} delta incoming 
 * @param {Int32Array} indices an array containing the index corresponding to the max pooled value
 * @param {Number} h height of the input tensor
 * @param {*} w width of the input tensor
 * @param {*} d depth of the input tensor
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns 
 */
const MaxPoolDelta = (delta, indices, h, w, d, modelID, layerID) => functions.MaxPoolDelta(
    delta, 
    indices, 
    h, 
    w, 
    d,
    modelID,
    layerID
);


/**
 * "✅☑️"
 * @param {Float32Array} input input vector
 * @param {Float32Array} prevHiddenState hidden temporal state
 * @param {Array<Number>} inputWeightShape input weight shape
 * @param {Array<Number>} recurrentWeightShape recurrent weight shape
 * @param {Number} pointer value to reference the weights and biases
 * @param {String} modelID model ID 
 * @returns 
 */
const recurrentMatMul = (input, prevHiddenState, inputWeightShape, recurrentWeightShape, pointer, modelID) => functions.recurrentMatMul(
    input, 
    prevHiddenState,
    inputWeightShape, 
    recurrentWeightShape, 
    getGlobalParams(modelID).globalWeights[pointer], 
    getGlobalParams(modelID).globalBiases[pointer],
    modelID
);

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Array<Number>} inputWeightShape 
 * @param {Array<Number>} recurrentWeightShape 
 * @param {Number} pointer 
 * @param {String} modelID model ID
 * @returns 
 */
const recurrentTimeDelta = (input, inputWeightShape, recurrentWeightShape, pointer, modelID) => functions.recurrentTimeDelta(
    input, 
    inputWeightShape,
    recurrentWeightShape,
    getGlobalParams(modelID).globalWeights[pointer],
    modelID
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
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns {Float32Array}
 */
const gradientClipping = (grads, threshold, modelID, layerID) => functions.gradientClipping(
    grads, 
    threshold,
    modelID, 
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Array<Number>} inputShape 
 * @param {Array<Number>} outputShape 
 * @param {Number} strides 
 * @param {Number} filters 
 * @param {Array<Number>} weightShape 
 * @param {Number} pointer 
 * @param {String} modelID model ID
 * @param {String} layerID 
 * @returns {Float32Array} trans conv output.
 */
const transConv = (input, inputShape, outputShape, strides, filters, weightShape, pointer, modelID, layerID) => functions.transConv(
    input, 
    inputShape, 
    outputShape, 
    strides, 
    filters, 
    weightShape, 
    getGlobalParams(modelID).globalWeights[pointer], 
    getGlobalParams(modelID).globalBiases[pointer],
    pointer,
    modelID,
    layerID
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
 * @param {String} modelID
 * @param {String} layerID
 * @returns {Float32Array}
 */
const transConvBackward = (input, inputShape, outputShape, strides, filters, weightShape, pointer, modelID, layerID) => functions.transConvBackward(
    input,
    inputShape,
    outputShape,
    strides,
    filters,
    weightShape,
    getGlobalParams(modelID).globalWeights[pointer],
    pointer,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} activations 
 * @param {Float32Array} deltas 
 * @param {Float32Array} weightGrads 
 * @param {Float32Array} biasGrads 
 * @param {Array<Number>} inputShape 
 * @param {Array<Number>} outputShape 
 * @param {Array<Number>} weightShape 
 * @param {Number} strides 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns {{ weightGrads: Float32Array, biasGrads: Float32Array }}
 */
const accumulateWeightandBiasGradsForTransConv = (activations, deltas, weightGrads, biasGrads, inputShape, outputShape, weightShape, strides, modelID, layerID) => functions.accumulateWeightandBiasGradsForTransConv(
    activations,
    deltas,
    weightGrads,
    biasGrads,
    inputShape, 
    outputShape,
    weightShape,
    strides,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {number} size 
 * @param {number} eps
 * @param {number} pointer 
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns 
 */
const computeLayerNorm = (input, size, eps, pointer, modelID, layerID) => functions.computelayerNorm(
    input, 
    size, 
    getGlobalParams(modelID).globalWeights[pointer], 
    getGlobalParams(modelID).globalBiases[pointer], 
    eps,
    pointer,
    modelID,
    layerID
);

/**
 * 
 * @param {Float32Array} dY incoming delta 
 * @param {Float32Array} X cached input during forward pass 
 * @param {Number} size feature size 
 * @param {number} pointer 
 * @param {String} modelID
 * @param {String} layerID 
 * @returns {{ dX: Float32Array, dGamma: Float32Array, dBeta: Float32Array }}
 */
const computeLayerNormBackward = (dY, X, size, eps, pointer, modelID, layerID) => functions.computeLayerNormBackward(
    dY,
    X,
    getGlobalParams(modelID).globalWeights[pointer],
    size, 
    eps,
    pointer,
    modelID,
    layerID
);

/**
 * 
 * @param {*} flat_arr_1 
 * @param {*} flat_arr_2 
 * @param {*} flat_arr_3 
 * @param {*} pointer 
 * @param {*} modelID 
 * @returns 
 */
const accumulate_element_wise_mul = (flat_arr_1, flat_arr_2, flat_arr_3, pointer, modelID) => {

    if (!flat_arr_1 || !flat_arr_2 || !flat_arr_3) throw new Error("[ERROR]------- requires '3' input arrays for this operation."); 

    if (flat_arr_1.length !== flat_arr_2.length || flat_arr_1.length !== flat_arr_3.length) throw new Error(`${red}[ERROR]------- Error: 3 input arrays are not equal in length. array1: ${flat_arr_1.length} | array2: ${flat_arr_2.length} ${reset} | array3: ${flat_arr_3.length}`);

    return functions.accumulate_element_wise_mul(flat_arr_1, flat_arr_2, flat_arr_3, pointer, modelID);
};

/**
 * "✅☑️"
 * @param {Float32Array} input the input tensor
 * @param {Number} embedDim embedding dimension
 * @param {Number} seqLen sequence length value 
 * @param {Number} dkRoot dkRoot value. Used for scaling attention scores
 * @param {Number} pointer pointer value to reference corresponding layer parameter 
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns {{ X: Float32Array, Q: Float32Array, K: Float32Array, V: Float32Array, S: Float32Array, output: Float32Array}}
 */
const CoreAttention = (input, embedDim, seqLen, dkRoot, pointer, modelID, layerID) => functions.CoreAttention(
    input, 
    getGlobalParams(modelID).globalWeights[pointer],
    getGlobalParams(modelID).globalBiases[pointer],
    embedDim,
    seqLen,
    dkRoot,
    pointer,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} delta incoming delta
 * @param {Float32Array} Q cached Q
 * @param {Float32Array} K cached K
 * @param {Float32Array} V cached V
 * @param {Float32Array} S cached softmax outputs
 * @param {Number} embedDim embedding dimension
 * @param {Number} seqLen sequence length value 
 * @param {Number} dkRoot dkRoot value. Used for scaling attention delta scores
 * @param {Number} pointer pointer value to reference corresponding layer parameter 
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns {{dQ: Float32Array, dK: Float32Array, dV: Float32Array, dX: Float32Array}}
 */
const CoreAttentionBackward = (delta, Q, K, V, S, embedDim, seqLen, dkRoot, pointer, modelID, layerID) => functions.CoreAttentionBackward(
    delta, 
    Q, 
    K,
    V, 
    S,
    getGlobalParams(modelID).globalWeights[pointer],
    embedDim,
    seqLen,
    dkRoot,
    pointer,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} input 
 * @param {Number} embedDim embedding dimension
 * @param {Number} seqLen sequence length value 
 * @param {Number} numHeads number of heads that process scaled dot product in parallel
 * @param {Number} headDim head dim value
 * @param {Number} dkRoot dkRoot value. Used for scaling attention scores
 * @param {Boolean} useCausalMasking causal masking state. If set to `true`, it will apply casual masking on the attention scores in order to not look up to future tokens.
 * @param {Number} pointer pointer value to reference corresponding layer parameter 
 * @param {String} modelID string value to reference model's unique parameters
 * @param {String} layerID string value to reference model's unique parameters
 * @returns {{X: Float32Array, Q: Float32Array, K: Float32Array, V: Float32Array, mhaOutput: Float32Array, S_perHead: Float32Array, finalOutput: Float32Array}}
 */
const CoreMultiHeadAttention = (input, embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking = false, pointer, modelID, layerID) => functions.CoreMultiHeadAttention(
    input,
    getGlobalParams(modelID).globalWeights[pointer],
    getGlobalParams(modelID).globalBiases[pointer],
    embedDim,
    seqLen, 
    numHeads,
    headDim, 
    dkRoot,
    useCausalMasking,
    pointer,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} delta incoming delta
 * @param {Float32Array} Q cached Q
 * @param {Float32Array} K cached K
 * @param {Float32Array} V cached V
 * @param {Float32Array} S cached softmax output per head, flattened head-major
 * @param {Number} embedDim embedding dimension
 * @param {Number} seqLen sequence length value 
 * @param {Number} numHeads number of heads that process scaled dot product in parallel
 * @param {Number} headDim head dim value
 * @param {Number} dkRoot dkRoot value. Used for scaling attention delta scores
 * @param {Boolean} useCasualMasking casual masking state. If set to `true`, it will apply casual masking on the attention scores in order to not look up to future tokens.
 * @param {Number} pointer pointer value to reference corresponding layer parameter 
 * @param {String} modelID string value to reference model's unique parameters
 * @param {String} layerID string value to reference model's unique parameters
 * @returns {{dQ: Float32Array, dK: Float32Array, dV: Float32Array, dMhaOutput: Float32Array, dX: Float32Array}}
 */
const CoreMultiHeadAttentionBackward = (delta, Q, K, V, S, embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking = false, pointer, modelID) => functions.CoreMultiHeadAttentionBackward(
    delta,
    getGlobalParams(modelID).globalWeights[pointer],
    Q,
    K,
    V,
    S,
    embedDim,
    seqLen,
    numHeads,
    headDim,
    dkRoot,
    useCausalMasking,
    pointer,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} dQ 
 * @param {Float32Array} dK 
 * @param {Float32Array} dV 
 * @param {Float32Array} deltaMHA 
 * @param {Float32Array} MHA_output 
 * @param {Float32Array} activation_outputs 
 * @param {Float32Array} weightGrads 
 * @param {Number} embedDim 
 * @param {Number} seqLen 
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns 
 */
const accumulateAttentionWeightsGradients = (dQ, dK, dV, deltaMHA, MHA_output, activation_outputs, weightGrads, embedDim, seqLen, modelID, layerID) => functions.accumulateAttentionWeightsGradients(
    dQ,
    dK,
    dV,
    deltaMHA,
    MHA_output,
    activation_outputs,
    weightGrads,
    embedDim,
    seqLen,
    modelID, 
    layerID
);


/**
 * "✅☑️"
 * @param {Float32Array} dQ 
 * @param {Float32Array} dK 
 * @param {Float32Array} dV 
 * @param {Float32Array} dMhaOutput 
 * @param {Float32Array} biasGrads 
 * @param {Number} embedDim 
 * @param {Number} seqLen 
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns 
 */
const accumulateAttentionBiasGrads = (dQ, dK, dV, dMhaOutput, biasGrads, embedDim, seqLen,  modelID, layerID) => functions.accumulateAttentionBiasGrads(
    dQ,
    dK,
    dV,
    dMhaOutput,
    biasGrads,
    embedDim,
    seqLen,
    modelID, 
    layerID
);

/**
 * "☑️"
 * @param {Float32Array} dQ 
 * @param {Float32Array} dK 
 * @param {Float32Array} dV 
 * @param {Float32Array} activation_outputs 
 * @param {Float32Array} weightGrads 
 * @param {Number} embedDim 
 * @param {Number} seqLen 
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns 
 */
const accumulateSimpleAttentionWeightGrads = (dQ, dK, dV, activation_outputs, weightGrads, embedDim, seqLen, modelID, layerID) => functions.accumulateSimpleAttentionWeightGrads(
    dQ,
    dK,
    dV,
    activation_outputs,
    weightGrads,
    embedDim,
    seqLen,
    modelID,
    layerID
);

/**
 * "☑️"
 * @param {Float32Array} dQ 
 * @param {Float32Array} dK 
 * @param {Float32Array} dV 
 * @param {Float32Array} biasGrads 
 * @param {Number} embedDim 
 * @param {Number} seqLen 
 * @param {String} modelID model ID
 * @param {String} layerID layerID
 * @returns 
 */
const accumulateSimpleAttentionBiasGrads = (dQ, dK, dV, biasGrads, embedDim, seqLen, modelID, layerID) => functions.accumulateSimpleAttentionBiasGrads(
    dQ,
    dK,
    dV,
    biasGrads,
    embedDim,
    seqLen,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} gammaGrads 
 * @param {Float32Array} dGamma 
 * @param {Float32Array} betaGrads 
 * @param {Float32Array} dBeta
 * @param {String} modelID 
 * @param {String} layerID 
 * @returns {{gammaGrads: Float32Array, betaGrads: Float32Array}}
 */
const AccumulateGammaAndBetaGrads = (gammaGrads, dGamma, betaGrads, dBeta, modelID, layerID) => functions.AccumulateGammaAndBetaGrads(
    gammaGrads,
    dGamma,
    betaGrads,
    dBeta,
    modelID,
    layerID
);

/**
 * "✅☑️"
 * @param {Float32Array} delta 
 * @param {String} modelID  
 * @param {String} layerID 
*/
const cacheOutputLayerDelta = (delta, modelID, layerID) => {

    if (addon && globalState().computeBackend === "opencl") addon.cacheOutputLayerDelta(delta, modelID, layerID);

}

module.exports = {
    init,
    shutdown,

    relu,
    sigmoid,
    tanh,
    softmax,
    linear,

    getEmbeddings,
    returnEmbeddings,
    sinusoidalPE,

    MatMul,
    DeltaMatMul,
    accumulateWeightsAndBiasGradsForConnectedLayer,
    
    ConvolveForward,
    ConvolveBackward,
    AccumulateWeightAndBiasGradsForConv,

    transConv,
    transConvBackward,
    accumulateWeightandBiasGradsForTransConv,

    element_wise_mul,
    element_wise_sub,
    element_wise_add,
    accumulate_element_wise_mul,
    scale,
    scaleDiff,

    ApplySGD,
    ApplyAdam,
    ApplyRMSProp,
    
    MaxPool,
    MaxPoolDelta,

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
    CoreAttentionBackward,
    accumulateSimpleAttentionWeightGrads,
    accumulateSimpleAttentionBiasGrads,

    CoreMultiHeadAttention,
    CoreMultiHeadAttentionBackward,
    accumulateAttentionWeightsGradients,
    accumulateAttentionBiasGrads,

    computeLayerNorm,
    computeLayerNormBackward,
    AccumulateGammaAndBetaGrads,

    cacheOutputLayerDelta,

    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
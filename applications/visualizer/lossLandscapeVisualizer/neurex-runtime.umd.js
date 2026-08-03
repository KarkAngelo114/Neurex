(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.NeurexRuntime = {}));
})(this, (function (exports) { 'use strict';

    var globals;
    var hasRequiredGlobals;

    function requireGlobals () {
    	if (hasRequiredGlobals) return globals;
    	hasRequiredGlobals = 1;
    	let globalWeights = []; // global array of weights
    	let globalBiases = []; // global array of biases
    	let globalOutputTensorTemplate = []; // global array of output templates used in feedforward only so that no need to create new Flaot32Array each time a layer function is called and to return an output during feedforward. Applies only to layers


    	const setGlobalParams = (weights, biases, outputTemplates) => {
    	    globalWeights = weights;
    	    globalBiases = biases;
    	    globalOutputTensorTemplate = outputTemplates;
    	};

    	/** 
    	 * Use to get paramters from the global store. 
    	 * @returns {{globalWeights: Array<Float32Array>, globalBiases: Array<Float32Array>, globalOutputTensorTemplate: Array<Float32Array>}}
    	*/
    	const getGlobalParams = () => {
    	    return {
    	        globalWeights: globalWeights,
    	        globalBiases: globalBiases,
    	        globalOutputTensorTemplate: globalOutputTensorTemplate
    	    }
    	};

    	globals = {
    	    setGlobalParams,
    	    getGlobalParams
    	};
    	return globals;
    }

    var float32Ops;
    var hasRequiredFloat32Ops;

    function requireFloat32Ops () {
    	if (hasRequiredFloat32Ops) return float32Ops;
    	hasRequiredFloat32Ops = 1;
    	const { getGlobalParams } = requireGlobals();

    	const Relu = (arr) => {
    	    const output = new Float32Array(arr);
    	    for (let i = 0; i < output.length; i++) {
    	        output[i] = output[i] > 0 ? output[i] : 0;
    	    }
    	    return output;
    	};

    	const Sigmoid = (arr) => {
    	    const output = new Float32Array(arr);
    	    for (let i = 0; i < output.length; i++) {
    	        output[i] = 1 / (1 + Math.exp(-output[i]));
    	    }
    	    return output;
    	};

    	const Tanh = (arr) => {
    	    const output = new Float32Array(arr);
    	    for (let i = 0; i < output.length; i++) {
    	        output[i] = Math.tanh(output[i]);
    	    }
    	    return output;
    	};

    	const Softmax = (arr) => {
    	    const output = new Float32Array(arr);
    	    const maxVal = Math.max(...output);
    	    let sum = 0;

    	    for (let i = 0; i < output.length; i++) {
    	        output[i] = Math.exp(output[i] - maxVal);
    	        sum += output[i];
    	    }

    	    for (let i = 0; i < output.length; i++) {
    	        output[i] /= sum;
    	    }

    	    return output;
    	};

    	const Linear = (arr) => {
    	    return new Float32Array(arr);
    	};

    	const DReLu = (arr) => {
    	    const output = new Float32Array(arr);
    	    for (let i = 0; i < output.length; i++) {
    	        output[i] = output[i] > 0 ? 1 : 0;
    	    }
    	    return output;
    	};

    	const DSigmoid = (arr) => {
    	    const output = new Float32Array(arr);
    	    for (let i = 0; i < output.length; i++) {
    	        const s = 1 / (1 + Math.exp(-output[i]));
    	        output[i] = s * (1 - s);
    	    }
    	    return output;
    	};

    	const DTanh = (arr) => {
    	    const output = new Float32Array(arr);
    	    for (let i = 0; i < output.length; i++) {
    	        const t = Math.tanh(output[i]);
    	        output[i] = 1 - t * t;
    	    }
    	    return output;
    	};

    	const DSoftmax = (arr) => {

    	    return new Float32Array(arr.length).fill(1);
    	};

    	const DLinear = (arr) => {
    	    const output = new Float32Array(arr.length);
    	    output.fill(1);
    	    return output;
    	};

    	const getEmbeddings = (tokenVector, embeddingDim, lookup, outputTemplatePointer) => {
    	    const {globalOutputTensorTemplate} = getGlobalParams();

    	    const output = globalOutputTensorTemplate[outputTemplatePointer];

    	    // helper function
    	    const getRow = (tokenID) => {
    	        const start = tokenID * embeddingDim;

    	        return lookup.subarray(start, start + embeddingDim);
    	    };

    	    const sequence_length = tokenVector.length;

    	    for (let i = 0; i < sequence_length; i++) {
    	        const row = getRow(tokenVector[i]);

    	        output.set(row, i * embeddingDim);
    	    }

    	    return output;
    	};

    	const MatMul = (input, inputSize, outputSize, weights, biases, outputTemplatePointer) => {
    	    const {globalOutputTensorTemplate} = getGlobalParams();
    	    
    	    const z_values = globalOutputTensorTemplate[outputTemplatePointer]; // use the output template pointer to get the corresponding pre-allocated output tensor

    	    // 1. Initialize with Biases (Faster than adding them in a separate loop later)
    	    z_values.set(biases);

    	    // 2. Perform Weighted Sum
    	    // We iterate through each input neuron
    	    for (let i = 0; i < inputSize; i++) {
    	        const inputVal = input[i];
    	        
    	        // Calculate the starting offset for this specific input neuron's weights
    	        const offset = i * outputSize;

    	        // Multiply the input by every weight connecting to output neurons
    	        for (let j = 0; j < outputSize; j++) {
    	            z_values[j] += inputVal * weights[offset + j];
    	        }
    	    }

    	    return z_values;
    	};

    	const ApplyPadding = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => {
    	    const newH = inputH + padTop + padBottom;
    	    const newW = inputW + padLeft + padRight;
    	    const output = new Float32Array(newH * newW * channels);

    	    for (let i = 0; i < inputH; i++) {
    	        for (let j = 0; j < inputW; j++) {
    	            for (let c = 0; c < channels; c++) {
    	                const oldIdx = (i * inputW + j) * channels + c;
    	                const newIdx = ((i + padTop) * newW + (j + padLeft)) * channels + c;
    	                output[newIdx] = input[oldIdx];
    	            }
    	        }
    	    }
    	    return {
    	        data: output,
    	        shape: [newH, newW, channels]
    	    };
    	};

    	const Convolve = (input, strides, outputShape, kernelShape, inputShape, weights, biases) => {

    	    const [numFilters, kernelH, kernelW, depth] = kernelShape;
    	    const [inputH, inputW] = inputShape;
    	    const [outputH, outputW] = outputShape;

    	    const output = new Float32Array(outputH * outputW * numFilters);

    	    const kernelSize = kernelH * kernelW * depth;

    	    for (let y = 0; y < outputH; y++) {

    	        const baseY = y * strides;

    	        for (let x = 0; x < outputW; x++) {

    	            const baseX = x * strides;

    	            const outBase = (y * outputW + x) * numFilters;

    	            for (let f = 0; f < numFilters; f++) {

    	                let sum = biases[f];

    	                const filterOffset = f * kernelSize;

    	                for (let ky = 0; ky < kernelH; ky++) {

    	                    const inY = baseY + ky;

    	                    if (inY >= inputH) continue;

    	                    for (let kx = 0; kx < kernelW; kx++) {

    	                        const inX = baseX + kx;

    	                        if (inX >= inputW) continue;

    	                        const inputBase = (inY * inputW + inX) * depth;

    	                        const kernelBase = filterOffset + (ky * kernelW + kx) * depth;

    	                        let c = 0;

    	                        for (; c <= depth - 4; c += 4) {
    	                            sum += input[inputBase + c] * weights[kernelBase + c];
    	                            sum += input[inputBase + c + 1] * weights[kernelBase + c + 1];
    	                            sum += input[inputBase + c + 2] * weights[kernelBase + c + 2];
    	                            sum += input[inputBase + c + 3] * weights[kernelBase + c + 3];
    	                        }

    	                        for (; c < depth; c++) {
    	                            sum += input[inputBase + c] * weights[kernelBase + c];
    	                        }
    	                    }
    	                }

    	                output[outBase + f] = sum;
    	            }
    	        }
    	    }

    	    return output;
    	};

    	const DilateInput = (input, shape, stride) => {
    	    const [H, W, C] = shape;
    	    const dilatedH = (H - 1) * stride + 1;
    	    const dilatedW = (W - 1) * stride + 1;
    	    
    	    const dilatedSize = dilatedH * dilatedW * C;
    	    const dilated = new Float32Array(dilatedSize);

    	    for (let c = 0; c < C; c++) {
    	        for (let h = 0; h < H; h++) {
    	            for (let w = 0; w < W; w++) {
    	                const srcIdx = (h * W + w) * C + c;
    	                const dilatedHIdx = h * stride;
    	                const dilatedWIdx = w * stride;
    	                const dstIdx = (dilatedHIdx * dilatedW + dilatedWIdx) * C + c;
    	                dilated[dstIdx] = input[srcIdx];
    	            }
    	        }
    	    }

    	    return {
    	        data: dilated,
    	        dilatedHeight: dilatedH,
    	        dilatedWidth: dilatedW
    	    };
    	};

    	const MaxPooling = (arr, pool_size, inputShape, outputShape, strides, outputTemplatePointer) => {
    	    const {globalOutputTensorTemplate} = getGlobalParams();
    	    const [poolH, poolW] = pool_size;
    	    const [inputH, inputW, inputD] = inputShape;
    	    const [outputH, outputW, outputD] = outputShape;

    	    const output = globalOutputTensorTemplate[outputTemplatePointer];
    	    const maxIdexes = new Int32Array(outputH * outputW * outputD);

    	    for (let d = 0; d < inputD; d++) {
    	        for (let i = 0; i < outputH; i++) {
    	            for (let j = 0; j < outputW; j++) {
    	                let maxVal = -Infinity;
    	                let maxIdx = -1;
    	                // Define the window boundaries based on strides
    	                const startH = i * strides;
    	                const startW = j * strides;

    	                for (let ph = 0; ph < poolH; ph++) {
    	                    for (let pw = 0; pw < poolW; pw++) {
    	                        const currH = startH + ph;
    	                        const currW = startW + pw;

    	                        // Check bounds to handle cases where window might exceed input dimensions
    	                        if (currH < inputH && currW < inputW) {
    	                            // Calculate index in the flattened 1D array
    	                            const idx = (currH * inputW * inputD) + (currW * inputD) + d;
    	                            const val = arr[idx];
    	                            if (val > maxVal) {
    	                                maxVal = val;
    	                                maxIdx = idx;
    	                            }	                        }
    	                    }
    	                }
    	                // Set the max value in the output array
    	                const outIdx = (i * outputW * outputD) + (j * outputD) + d;
    	                output[outIdx] = maxVal === -Infinity ? 0 : maxVal;
    	                maxIdexes[outIdx] = maxIdx;
    	            }
    	        }
    	    }
    	    return {
    	        output: output,
    	        maxIndices: maxIdexes
    	    };
    	};

    	const recurrentMatMul = (input, prevHiddenState,  inputWeightShape, recurrentWeightShape, weights, biases, outputTemplatePointer) => {
    	    const { globalOutputTensorTemplate } = getGlobalParams();
    	    // The weights were concatenated during initialization as:
    	    // [input_weights..., recurrent_weights...]
    	    const inputSize = inputWeightShape[0];
    	    const units = inputWeightShape[1];
    	    const range_input_weights = inputSize * units;
    	    const output = globalOutputTensorTemplate[outputTemplatePointer];

    	    const input_weights = weights.subarray(0, range_input_weights);
    	    const recurrent_weights = weights.subarray(range_input_weights, range_input_weights + recurrentWeightShape[0] * recurrentWeightShape[1]);

    	    for (let j = 0; j < units; j++) {
    	        let z = biases[j];

    	        for (let i = 0; i < inputSize; i++) {
    	            z += input[i] * input_weights[i * units + j];
    	        }

    	        for (let h = 0; h < units; h++) {
    	            z += prevHiddenState[h] * recurrent_weights[h * units + j];
    	        }

    	        output[j] = z;
    	    }

    	    return output;
    	};


    	float32Ops = {
    	    Relu,
    	    Sigmoid,
    	    Tanh,
    	    Softmax,
    	    Linear,
    	    DReLu,
    	    DSigmoid,
    	    DTanh,
    	    DSoftmax,
    	    DLinear,
    	    getEmbeddings,
    	    MatMul,
    	    ApplyPadding,
    	    Convolve,
    	    DilateInput,
    	    Convolve,
    	    MaxPooling,
    	    recurrentMatMul,
    	};
    	return float32Ops;
    }

    var entry;
    var hasRequiredEntry;

    function requireEntry () {
    	if (hasRequiredEntry) return entry;
    	hasRequiredEntry = 1;
    	const { getGlobalParams } = requireGlobals();

    	/**
    	  * this runtime version only needs feedforward related functions.
    	  * No need to use native bindings
    	  */

    	const functions = requireFloat32Ops();

    	/**
    	 *  "✅☑️"
    	 * @function getEmbeddings
    	 * @param {Array<Number>} tokenVector an array of token vector 
    	 * @param {Number} embeddingDim embedding dim value
    	 * @param {Number} pointer pointer value corresponding to the global parameter of weights and biases 
    	 * @param {Number} outputTemplatePointer pointer value correspondind to the output template tensor 
    	 * @returns {Float32Array} flattened embeddings
    	 */
    	const getEmbeddings = (tokenVector, embeddingDim, pointer, outputTemplatePointer) => functions.getEmbeddings(
    	    Array.from(tokenVector), 
    	    embeddingDim, 
    	    getGlobalParams().globalWeights[pointer],
    	    outputTemplatePointer
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
    	 * @function relu
    	 * @param {Float32Array} input - 1D array of features 
    	 * @returns - 1D array of activated features (Using ReLu)
    	 */
    	const relu = (input) => functions.Relu(input);

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
    	const dsoftmax = (input) => functions.DSoftmax(input);

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
    	 * @function MaxPool
    	 * @param {Float32Array} input - current input passed down to this layer 
    	 * @param {Array<Number>} poolSize - pool size of the sliding window
    	 * @param {Array<Number>} inputShape - input shape of the current tensor
    	 * @param {Array<Number>} outputShape - output shape of the tensor
    	 * @param {Number} strides - determines how many pixels it will skipped
    	 */
    	const MaxPool = (input, poolSize, inputShape, outputShape, strides, outputTemplatePointer) => functions.MaxPooling(input, poolSize, inputShape, outputShape, strides, outputTemplatePointer);

    	/**
    	 * "☑️"
    	 * @param {Float32Array} input input vector
    	 * @param {Float32Array} prevHiddenState hidden temporal state
    	 * @param {Array<Number>} inputWeightShape input weight shape
    	 * @param {Array<Number>} recurrentWeightShape recurrent weight shape
    	 * @param {Number} pointer value to reference the weights and biases 
    	 * @param {Number} outputTemplatePointer value to reference the output template pointer 
    	 * @returns 
    	 */
    	const recurrentMatMul = (input, prevHiddenState, inputWeightShape, recurrentWeightShape, pointer, outputTemplatePointer) => functions.recurrentMatMul(
    	    input, 
    	    prevHiddenState,
    	    inputWeightShape, 
    	    recurrentWeightShape, 
    	    getGlobalParams().globalWeights[pointer], 
    	    getGlobalParams().globalBiases[pointer],
    	    outputTemplatePointer
    	);


    	entry = {
    	    getEmbeddings,
    	    MatMul,
    	    relu,
    	    sigmoid,
    	    tanh,
    	    softmax,
    	    linear,
    	    applyPadding,
    	    Convolve,
    	    Dilate_Input,
    	    MaxPool,
    	    recurrentMatMul,
    	    derivatives: {
    	        relu: drelu,
    	        sigmoid: dsigmoid,
    	        tanh: dtanh,
    	        softmax: dsoftmax,
    	        linear: dlinear
    	    },
    	};
    	return entry;
    }

    var inputLayer;
    var hasRequiredInputLayer;

    function requireInputLayer () {
    	if (hasRequiredInputLayer) return inputLayer;
    	hasRequiredInputLayer = 1;
    	/**
    	 * 
    	 * @param {Object} shapeConfig shape configuration object to set the size and shape the network is expecting
    	 * @returns {Object} input layer config 
    	 */
    	const inputConfig = (shapeConfig) => {
    	    try {
    	        if (shapeConfig.featureSize && shapeConfig.sequenceLength) {
    	            const { featureSize, sequenceLength } = shapeConfig;
    	            return {
    	                layer_name: "input_layer",
    	                layer_size: featureSize * sequenceLength,
    	                input_shape: [1, 1, featureSize, sequenceLength]
    	            };
    	        }
    	        else if (shapeConfig.features) {
    	            const features = shapeConfig.features;
    	            return {
    	                layer_name: "input_layer",
    	                layer_size: features,
    	                input_shape: [1, 1, features, 1]
    	            };
    	        }
    	        else if (shapeConfig.height && shapeConfig.width && shapeConfig.depth) {
    	            const { height, width, depth } = shapeConfig;

    	            return {
    	                layer_name: "input_layer",
    	                layer_size: height * width * depth,
    	                input_shape: [height, width, depth]
    	            };
    	        } 
    	        else {
    	            throw new Error(`[ERROR]------- Invalid input shape config`);
    	        }
    	    } 
    	    catch (error) {
    	        console.error(error.message);
    	    }
    	};

    	inputLayer = inputConfig;
    	return inputLayer;
    }

    var utils;
    var hasRequiredUtils;

    function requireUtils () {
    	if (hasRequiredUtils) return utils;
    	hasRequiredUtils = 1;
    	const XavierInitialization = (inputSize, outputSize) => {
    	    return Math.sqrt(2 / (inputSize + outputSize));
    	};

    	const calculateTensorShape = (inputHeight, inputWidth, kernelHeight, kernelWidth, depth, stride, padding) => {
    	    // console.log(inputHeight, inputWidth, kernelHeight, kernelWidth, depth, stride, padding);
    	    let oH, oW;
    	    if (padding === "same") {
    	        oH = Math.ceil(inputHeight / stride);
    	        oW = Math.ceil(inputWidth / stride);
    	    } else {
    	        oH = Math.floor((inputHeight - kernelHeight) / stride + 1);
    	        oW = Math.floor((inputWidth - kernelWidth) / stride + 1);
    	    }

    	    return {
    	        OutputHeight: oH,
    	        OutputWidth: oW,
    	        CalculatedTensorShape: oH * oW * depth
    	    };
    	};

    	/**
    	 * 
    	 * @param {Number} inputH - height of the input
    	 * @param {Number} inputW - width of the input 
    	 * @param {Number} kernelH - height of the kernel
    	 * @param {Number} kernelW - width of the kernel
    	 * @param {Number} stride - stride value
    	 * @param {String} padding - "same" or "valid"
    	 * @returns 
    	 */
    	const getPaddingSizes = (inputH, inputW, kernelH, kernelW, stride, padding) => {
    	    if (padding === "valid") {
    	        return { top: 0, bottom: 0, left: 0, right: 0 };
    	    }

    	    // Standard formula for total padding needed
    	    const outputH = Math.ceil(inputH / stride);
    	    const outputW = Math.ceil(inputW / stride);

    	    const padH = Math.max(0, (outputH - 1) * stride + kernelH - inputH);
    	    const padW = Math.max(0, (outputW - 1) * stride + kernelW - inputW);

    	    // Distribute padding to sides (asymmetric if necessary)
    	    return {
    	        top: Math.floor(padH / 2),
    	        bottom: padH - Math.floor(padH / 2),
    	        left: Math.floor(padW / 2),
    	        right: padW - Math.floor(padW / 2)
    	    };
    	};

    	const ifOneHotEndcoded = (Y_train) => {
    	        /**
    	        Checks if all rows in Y_train are one-hot encoded.
    	        Each row must:
    	        - Contain only 0s and 1s
    	        - Have exactly one "1"
    	        */
    	        for (let i = 0; i < Y_train.length; i++) {
    	            const row = Y_train[i];
    	            if (!Array.isArray(row)) return false;

    	            let onesCount = 0;
    	            for (let j = 0; j < row.length; j++) {
    	                if (row[j] !== 0 && row[j] !== 1) return false;
    	                if (row[j] === 1) onesCount++;
    	            }

    	            if (onesCount !== 1) return false;
    	        }
    	        return true;
    	    };

    	const getTotalMB = (array) => {
    	    let sum = 0;
    	    for (let i = 0; i < array.length; i++) {
    	        sum += array[i].byteLength / (1024 * 1024);
    	    }
    	    return sum;
    	};

    	const formatDuration = (totalSeconds) => {
    	    const d = Math.floor(totalSeconds / (3600 * 24));
    	    const h = Math.floor((totalSeconds % (3600 * 24)) / 3600);
    	    const m = Math.floor((totalSeconds % 3600) / 60);
    	    const s = totalSeconds % 60; 

    	    const parts = [];
    	    if (d > 0) parts.push(`${d}d`);
    	    if (h > 0) parts.push(`${h}h`);
    	    if (m > 0) parts.push(`${m}m`);
    	    
    	    // Use .toFixed(1) for one decimal place (e.g., 0.2s)
    	    if (s > 0 || parts.length === 0) {
    	        parts.push(`${s.toFixed(3)}s`);
    	    }

    	    return parts.join(' ');
    	};

    	/**
    	 * 
    	 * @param {Array<Float32Array>} chunks an array collection of float32 array 
    	 * @returns { Float32Array }
    	 */
    	const concatenateFloat32Array = (chunks) => {
    	    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    	    const result = new Float32Array(totalLength);
    	    
    	    // 2. Copy each chunk into the target position
    	    let offset = 0;
    	    for (const chunk of chunks) {
    	        result.set(chunk, offset);
    	        offset += chunk.length;
    	    }
    	    
    	    return result;
    	};

    	utils = {
    	    calculateTensorShape,
    	    getPaddingSizes,
    	    XavierInitialization,
    	    ifOneHotEndcoded,
    	    getTotalMB,
    	    formatDuration,
    	    concatenateFloat32Array
    	};
    	return utils;
    }

    var connectedLayer;
    var hasRequiredConnectedLayer;

    function requireConnectedLayer () {
    	if (hasRequiredConnectedLayer) return connectedLayer;
    	hasRequiredConnectedLayer = 1;
    	const { MatMul } = requireEntry();
    	requireUtils();
    	const activation = requireEntry();

    	/**
    	 * The feedforward logic of this layer
    	 * @param {Float32Array} input input features 
    	 * @param {Object} current_layer current layer object coonfiguration
    	 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
    	 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
    	 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
    	 */
    	const feedforward = (input, current_layer, pointer, outputTemplatePointer) => {
    	    const [inputSize, outputSize] = current_layer.weightShape; // weight shape [input, output]
    	    const z_values = MatMul(input, inputSize, outputSize, pointer, outputTemplatePointer); // perform the MatMul() operation

    	    const activation_function = activation[current_layer.activation_function.name]; // activation function
    	    let outputs = activation_function(z_values); // use the activation function       
    	    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");
    	                    
    	    return {
    	        outputs, 
    	        z_values,
    	        incrementor_value: 1
    	    };
    	};


    	connectedLayer = {
    	    feedforward
    	};
    	return connectedLayer;
    }

    var convolutionalLayer;
    var hasRequiredConvolutionalLayer;

    function requireConvolutionalLayer () {
    	if (hasRequiredConvolutionalLayer) return convolutionalLayer;
    	hasRequiredConvolutionalLayer = 1;
    	const activation = requireEntry();
    	const { applyPadding, Convolve, Dilate_Input} = requireEntry();
    	const { XavierInitialization, calculateTensorShape, getPaddingSizes } = requireUtils();

    	/**
    	 * The feedforward logic of this layer
    	 * @param {Float32Array} input input features 
    	 * @param {Object} current_layer current layer object coonfiguration
    	 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
    	 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
    	 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
    	 */
    	const feedforward = (input, current_layer, pointer, outputTemplatePointer) => {
    	    let [f, kh, kw, kd] = current_layer.weightShape;
    	    let [input_H, input_W, input_D] = current_layer.inputShape; 
    	    let padding = current_layer.padding;
    	    let strides = current_layer.strides;

    	    // 1. compute expected output tensor shape
    	    const { OutputHeight, OutputWidth } = calculateTensorShape(input_H, input_W, kh, kw, input_D, current_layer.strides, current_layer.padding);

    	    // 2. get padding sizes for each sides
    	    const {top, bottom, left, right} = getPaddingSizes(input_H, input_W, kh, kw, strides, padding);

    	    // 3. apply padding
    	    const {data, shape} = applyPadding(input, input_H, input_W, input_D, top, bottom, left, right);

    	    // 4. Perform the convolve operation using the shapes calculated in step 1
    	    const convolve_result = Convolve(data, current_layer.strides, [OutputHeight, OutputWidth], [f, kh, kw, kd], [shape[0], shape[1]], pointer, outputTemplatePointer);

    	    if (convolve_result.some(Number.isNaN)) throw new Error('NaN detected on convolve result');

    	    // 5. activate each depth input using the given activation function
    	    const activation_function = activation[current_layer.activation_function.name];
    	    const outputs = activation_function(convolve_result);

    	    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");

    	    return {
    	        outputs: outputs,
    	        z_values: convolve_result,
    	        incrementor_value: 1
    	    };
    	};

    	convolutionalLayer = {
    	    feedforward,
    	};
    	return convolutionalLayer;
    }

    var maxPooling;
    var hasRequiredMaxPooling;

    function requireMaxPooling () {
    	if (hasRequiredMaxPooling) return maxPooling;
    	hasRequiredMaxPooling = 1;
    	const { MaxPool } = requireEntry();

    	/**
    	 * The feedforward logic of this layer
    	 * @param {Float32Array} input input features 
    	 * @param {Object} current_layer current layer object coonfiguration
    	 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
    	 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
    	 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
    	 */
    	const feedforward = (input, current_layer, pointer, outputTemplatePointer) => {
    	    const [inputh, inputw, inputd] = current_layer.inputShape;
    	    const [outputh, outputw, outputd] = current_layer.outputShape;
    	    const [poolHeight, poolWidth] = current_layer.poolSize;
    	    const strides = current_layer.strides;
    	                
    	    let {output, maxIndices} = MaxPool(input, [poolHeight, poolWidth], [inputh, inputw, inputd], [outputh, outputw, outputd], strides, outputTemplatePointer);

    	    current_layer.maxIndices = maxIndices;

    	    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");

    	    return {
    	        outputs: output,
    	        z_values: output,
    	        incrementor_value:0
    	    }
    	};

    	maxPooling = {
    	    feedforward,
    	};
    	return maxPooling;
    }

    var embeddingLayer;
    var hasRequiredEmbeddingLayer;

    function requireEmbeddingLayer () {
    	if (hasRequiredEmbeddingLayer) return embeddingLayer;
    	hasRequiredEmbeddingLayer = 1;
    	const { getEmbeddings } = requireEntry();


    	/**
    	 * The feedforward logic of this layer
    	 * @param {Float32Array} input input features 
    	 * @param {Object} current_layer current layer object coonfiguration
    	 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
    	 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
    	 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
    	 */
    	const feedforward = (input, current_layer, pointer, outputTemplatePointer) => {
    	    const embeddingDim = current_layer.embeddingDim;

    	    const output = getEmbeddings(input, embeddingDim, pointer, outputTemplatePointer);

    	    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs on Embedding layer (feedforward)");
    	    
    	    return {
    	        outputs: output, 
    	        z_values: output,
    	        incrementor_value: 1
    	    };
    	};

    	embeddingLayer = {
    	    feedforward,
    	};
    	return embeddingLayer;
    }

    var recurrentCell;
    var hasRequiredRecurrentCell;

    function requireRecurrentCell () {
    	if (hasRequiredRecurrentCell) return recurrentCell;
    	hasRequiredRecurrentCell = 1;
    	const { recurrentMatMul } = requireEntry();
    	const activation = requireEntry();
    	const { concatenateFloat32Array } = requireUtils();

    	/**
    	 * The feedforward logic of this layer
    	 * @param {Float32Array} inputSequence input sequence data 
    	 * @param {Object} current_layer current layer object coonfiguration
    	 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
    	 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
    	 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
    	 */
    	const feedforward = (inputSequence, current_layer, pointer, outputTemplatePointer) => {

    	    const units = current_layer.units;
    	    // Assume inputSequence is flat: [units * sequence_length]
    	    const sequence_length = current_layer.maxSequenceLength || 1; 
    	    const feature_size = current_layer.weightShape[0]; // [feature_size/size, units]

    	    // 1. Initialize a clean hidden state vector for the start of THIS sequence sample
    	    let current_hidden = new Float32Array(units).fill(0); 

    	    // 2. Arrays to store history for Backpropagation Through Time (BPTT)
    	    const all_z_values = [];
    	    const all_hidden_states = []; 

    	    // 3. Loop through each time step sequentially
    	    for (let t = 0; t < sequence_length; t++) {
    	        
    	        // Extract x_t for the current time step
    	        let offset = t * feature_size;
    	        const sequence_data = inputSequence.subarray(offset, offset + feature_size);

    	        // Compute z_t = (x_t * W_x) + (current_hidden * W_h) + bias
    	        // Pass current_hidden explicitly so it doesn't leak between global samples
    	        const z_t = recurrentMatMul(
    	            sequence_data, 
    	            current_hidden, 
    	            [current_layer.weightShape[0], current_layer.weightShape[1]], 
    	            [current_layer.weightShape[2], current_layer.weightShape[3]], 
    	            pointer, 
    	            outputTemplatePointer
    	        );

    	        if (z_t.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs on Recurrent layer (feedforward)");
    	        
    	        // Update hidden state for the next step
    	        current_hidden = activation[current_layer.activation_function.name](z_t);

    	        // Record history for backprop
    	        all_z_values.push(z_t);
    	        all_hidden_states.push(new Float32Array(current_hidden));
    	    }

    	    // cache recurrent layer cell feedforward data
    	    current_layer.cache = {
    	        hidden_states: all_hidden_states,
    	        recurrentZs: all_z_values
    	    };

    	    let final_output;
    	    if (current_layer.return_sequence) {
    	        // Concatenate all hidden states into one big flat array if return_sequence is true
    	        final_output = concatenateFloat32Array(all_hidden_states);
    	    } else {
    	        // Just return the very last hidden state vector
    	        final_output = all_hidden_states[sequence_length - 1];
    	    }

    	    return {
    	        outputs: final_output, 
    	        z_values: all_z_values, // Pass the array of z_values back to Neurex
    	        incrementor_value: 1
    	    };
    	};

    	recurrentCell = {
    	    feedforward,
    	};
    	return recurrentCell;
    }

    /**
     * Neurex follows a Plugin-style architecture where in modifications on the core engine (the core file) are minimal and the logic are exposed by these methods of the Layers class.
     * This allows the library to be extensible, flexible, and clean separation of concern without touching the core engine
     * Read here about Plugin-style architecture: https://medium.com/omarelgabrys-blog/plug-in-architecture-dec207291800
     */

    var layers;
    var hasRequiredLayers;

    function requireLayers () {
    	if (hasRequiredLayers) return layers;
    	hasRequiredLayers = 1;
    	/**
    	  * this runtime version only needs feedforward related functions
    	  */

    	const activation = requireEntry();

    	// import modular functions of different layers. 
    	const inputConfig = requireInputLayer();
    	const ann = requireConnectedLayer();
    	const cnn = requireConvolutionalLayer();
    	const maxpool = requireMaxPooling();
    	const embedding = requireEmbeddingLayer();
    	const rnn = requireRecurrentCell();


    	class Layers {
    	    constructor () {
    	        this.weights = [];
    	        this.biases = [];
    	        this.weightGrads = [];
    	        this.biaeGrads = [];
    	    }

    	    /**
    	     * @method inputShape
    	     * @param {object} shapeConfig - specify the number of features
    	     * @returns {Object}
    	     * @example
    	     * model.sequentialBuild([
    	        layer.inputShape({features: 4}),
    	        layer.connectedLayer("relu", 5),
    	        layer.connectedLayer("softmax", 3);
    	     ]);

    	     the inputShape() method allows you to get the shape of your input.
    	     */
    	    inputShape = (shapeConfig) => inputConfig(shapeConfig);

    	    /**
    	    * Creates an embedding layer for token encoding.
    	    *
    	    * @param {Number} vocabSize - The size of the vocabulary.
    	    * @param {Number} embeddingDim - The size of the dense vector used to represent each token.
    	    * @param {Number} maxSequenceLength - The length of the encoded token containing token IDs.
    	    * @returns {Object} - The embedding layer object configuration
    	    */
    	    embeddingLayer(vocabSize, embeddingDim, maxSequenceLength) {
    	        if (vocabSize <= 0 || embeddingDim <= 0 || maxSequenceLength <= 0) throw new Error(`VocabSize or embeddingDim should not be a negative number or 0. vocabSize: ${vocabSize} | embeddingDim: ${embeddingDim} | maxSequenceLength: ${maxSequenceLength}`);

    	        return {
    	            layer_name:"EmbeddingLayer",
    	            vocabSize: vocabSize,
    	            embeddingDim: embeddingDim,
    	            maxSequenceLength: maxSequenceLength,
    	            feedforward: (input, current_layer, pointer, outputTemplatePointer) => embedding.feedforward(input, current_layer, pointer, outputTemplatePointer),
    	        }
    	    }

    	    /**
    	     * @method connectedLayer
    	     * @param {Number} layer_size specify the number of neuron for this layer. Default is `5`
    	     * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear, softmax). Default is `relu`.
    	     * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
    	     * @returns {Object}
    	     *
    	     * Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will
    	     * build connected layers or multilayer perceptron
    	     */
    	    connectedLayer(layer_size = 5, activation_function = 'relu') {
    	        try {

    	            if (!activation_function || !layer_size || layer_size <= 0) {
    	                throw new Error(`[ERROR]------- Layer Error | Activation function: ${activation_function} | layer size: ${layer_size}`);
    	            }

    	            let function_name = activation_function.toLowerCase();

    	            if (!activation[function_name] || !activation.derivatives[function_name]) {
    	                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
    	            }

    	            return {
    	                layer_name: "connected_layer", 
    	                activation_function: activation[function_name], 
    	                derivative_activation_function: activation.derivatives[function_name],
    	                layer_size: layer_size,
    	                feedforward: (input, current_layer, pointer, outputTemplatePointer) => ann.feedforward(input, current_layer, pointer, outputTemplatePointer),
    	            };
    	        }
    	        catch (error) {
    	            console.log(error.message);
    	        }
    	    }

    	    /**
    	     * 
    	     * @method convolutionalLayer
    	     * @param {Number} filters - the number of filters for this convolutional layer. Produces the same number of output features
    	     * @param {Number} strides - It determines how much the filter overlaps with the input as it slides across.
    	     * @param {Array<Number>} kernel_size - the size of the kernel (or filter) that will slide and extracts input features
    	     * @param {String} activation_function - the activation function to be use for this layer
    	     * @param {String} padding - adds extra values (typically 0s) around the border of an input before applying a convolutional filter
    	     * @throws {Error} - if any of the parameters are invalid.
    	     * @returns {Object}
    	     *
    	     * Allows you to add convolutional layers in your model architecture in sequential building.
    	     */
    	    convolutionalLayer(filters = 1, strides = 1, kernel_size = [3, 3], activation_function = 'relu', padding = 'same') {
    	        try {
    	            if (!filters || filters <= 0) throw new Error(`[ERROR]-------- Filters cannot be empty, less than or equal to 0. Filters: ${filters}`);
    	            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);
    	            if (!kernel_size || kernel_size.length == 0 || (kernel_size[0] <= 0 || kernel_size[1] <= 0)) throw new Error(`[ERROR]------- Kernels cannot be empty, nor it's height or width is less than or equal to 0. Kernel size: ${kernel_size}`);
    	            if (!activation_function || activation_function == undefined || activation_function == null || activation_function === "") throw new Error(`[ERROR]-------- activation_function cannot be empty, null or undefined.`);
    	            if (!padding || padding == undefined || padding == null || padding === "") throw new Error(`[ERROR]-------- Padding cannot be empty, null or undefined.`);

    	            // check if the padding is same/valid, otherwise throw error
    	            let paddings = ["same", "valid"];
    	            if (!paddings.includes(padding.toLowerCase())) {
    	                throw new Error(`[ERROR]------- ${padding.toLowerCase()} is invalid. Use 'same' or 'valid' only`);
    	            }

    	            // check if the activation function is valid
    	            const function_name = activation_function.toLowerCase();

    	            if (!activation[function_name] || !activation.derivatives[function_name]) {
    	                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
    	            }

    	            return {
    	                layer_name: "convolutionalLayer",
    	                activation_function: activation[function_name],
    	                derivative_activation_function: activation.derivatives[function_name],
    	                kernel_size: kernel_size,
    	                filters: filters,
    	                padding: padding.toLowerCase(),
    	                strides: strides,
    	                feedforward: (input, current_layer, pointer, outputTemplatePointer) => cnn.feedforward(input, current_layer, pointer, outputTemplatePointer),
    	            }
    	        }
    	        catch (error) {
    	            console.error(error);
    	            process.exit(1);
    	        }
    	    }

    	    /**
    	     * @method maxPooling
    	     * @param {Array<Number>} poolSize - determines the pool size window 
    	     * @param {Number} strides - It determines how much the pool window slides across the input tensor.
    	     * @param {String} padding - `same` or `valid`
    	     * @throws {Error} - if any of the values are 0s or negative for the pool size and strides or the padding is invalid
    	     *
    	     * `maxPooling` is use for downsampling operation that reduces the spatial dimensions of an input tensor by taking the maximum value over a defined sliding window
    	     */
    	    maxPooling(poolSize, strides = 1, padding = "same") {
    	        try {
    	            if (poolSize[0] <= 0 || poolSize[1] <= 0) {
    	                throw new Error(`[ERROR]------- pool size value cannot be 0 or a negative value`);
    	            }

    	            // check if the padding is same/valid, otherwise throw error
    	            let paddings = ["same", "valid"];
    	            if (!paddings.includes(padding.toLowerCase())) {
    	                throw new Error(`[ERROR]------- ${padding.toLowerCase()} is invalid. Use 'same' or 'valid' only`);
    	            }

    	            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);

    	            return {
    	                layer_name: "maxPooling",
    	                poolSize: poolSize,
    	                padding: padding,
    	                strides: strides,
    	                feedforward: (input, current_layer, pointer, outputTemplatePointer) => maxpool.feedforward(input, current_layer, pointer, outputTemplatePointer),
    	            }
    	        }
    	        catch (error) {
    	            console.error(error);
    	            process.exit(1);
    	        }
    	    }

    	    /**
    	     * 
    	     * @param {Number} units This is the number of hidden units (neurons) in the layer. It dictates the dimensionality of the layer's output space and its internal memory state. 
    	     * @param {String} activation_function The activation function applied to the internal hidden state. Default value is `tanh`.
    	     * @param {Boolean} return_sequence default value is `false`. If `false`, Outputs only the final hidden state vector at the very last time step. If set to `true`, Outputs the hidden state vector for every single time step in the sequence. Must be set to `true` if another RNN layer follows.
    	     * @param {Boolean} return_state default value is `false`. If `true`, the layer will return its final hidden state vector as a separate tensor alongside its standard output.
    	     */
    	    recurrentCell(units, activation_function = "tanh", return_sequence = false, return_state = false) {
    	        try {
    	            let function_name = activation_function.toLowerCase();

    	            if (!activation[function_name] || !activation.derivatives[function_name])  throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid.`);
    	            if (!units || units <= 0) throw new Error(`[ERROR]------- Units cannot be null, negative integer or a 0. | Units: ${units}`);

    	            return {
    	                layer_name: "recurrent_cell", 
    	                activation_function: activation[function_name], 
    	                derivative_activation_function: activation.derivatives[function_name],
    	                units: units,
    	                return_sequence: return_sequence,
    	                return_state: return_state,
    	                feedforward: (input, current_layer, pointer, outputTemplatePointer) => rnn.feedforward(input, current_layer, pointer, outputTemplatePointer),
    	            }
    	        }
    	        catch (error) {
    	            console.error(error);
    	            process.exit(1);
    	        }
    	    }
    	}

    	layers = Layers;
    	return layers;
    }

    var core;
    var hasRequiredCore;

    function requireCore () {
    	if (hasRequiredCore) return core;
    	hasRequiredCore = 1;
    	/**
    	  * this runtime version only needs feedforward related functions
    	  */

    	// import necessary modules
    	const Layers = requireLayers();
    	const { setGlobalParams } = requireGlobals();

    	class Runtime {
    	    constructor () {
    	        this.weights = [];
    	        this.biases = [];
    	        this.output_layers_templates = [];
    	        this.num_layers = 0;
    	        this.input_size = 1;
    	        this.input_shape = [1, 1, 1];
    	        this.output_shape = [];
    	        this.currentShape = null;
    	        this.currentSize = null;
    	        this.accuracy = '';
    	        this.loss_function = '';
    	        this.output_size = 0;
    	        this.task = null;
    	        this.epoch_count = 0;
    	        this.batch_size = 0;
    	        this.depth = 0;
    	        this.filters = 1;
    	        this.layers = []; // layers (except input type layers) and their details will store here
    	        this.hasSequentiallyBuild = false;
    	        this.hasBuilt = false;

    	        // default configs
    	        this.optimizer = 'sgd';
    	        this.learning_rate = 0.001;

    	        this.isfailed = false;

    	        this.parametric_layers = [];
    	        this.miscellaneous = null;
    	    }

    	    /**
    	     * 
    	     * @param {String} model - path to your model
    	     */
    	    async loadSavedModel(modelData) {
    	        try {
    	            if (!modelData) {
    	                throw new Error('[ERROR]------- No JSON model provided.');
    	            }

    	            if (this.layers.length > 0) {
    	                console.warn(`[WARN]------- A model is already loaded in this instance. If you're going to load another model, you have to instantiate another class`);
    	                return;
    	            }

    	            // Assign properties
    	            this.miscellaneous = modelData.miscellaneous;
    	            this.task = modelData.task;
    	            this.loss_function = modelData.loss_function;
    	            this.epoch_count = modelData.epoch;
    	            this.batch_size = modelData.batch_size;
    	            this.optimizer = modelData.optimizer;
    	            this.learning_rate = modelData.learning_rate;
    	            this.input_size = modelData.input_size;
    	            this.output_size = modelData.output_size;
    	            this.num_layers = modelData.num_layers;
    	            this.weights = modelData.weights.map(w => new Float32Array(w));
    	            this.biases = modelData.biases.map(b => new Float32Array(b));
    	            this.input_shape = modelData.input_shape;
    	            const layerBuilder = new Layers();
    	            this.layers = modelData.layers.map(layerData => {
    	                let newLayer;
    	                if (layerData.layer_name === "connected_layer") {
    	                    // Recreate the connected layer with the correct activation and size
    	                    newLayer = layerBuilder.connectedLayer(layerData.layer_size, layerData.activation_function_name);
    	                    newLayer.weightShape = layerData.weightShape;
    	                    this.output_layers_templates.push(new Float32Array(layerData.layer_size));
    	                    this.parametric_layers.push(layerData.layer_name);
    	                } else if (layerData.layer_name === "input_layer") {
    	                    // Recreate the input layer. Note: The input layer doesn't have methods, so this is just for consistency
    	                    newLayer = layerBuilder.inputShape({ features: layerData.layer_size });
    	                } else if (layerData.layer_name === "convolutionalLayer") {
    	                    // recreate Convolutional layer
    	                    newLayer = layerBuilder.convolutionalLayer(layerData.filters, layerData.strides, layerData.kernel_size, layerData.activation_function_name, layerData.padding);
    	                    newLayer.weightShape = layerData.weightShape;
    	                    newLayer.inputShape = layerData.inputShape;
    	                    newLayer.outputShape = layerData.outputShape;
    	                    const [H, W, D] = layerData.outputShape;
    	                    const totalSize = H * W * D;
    	                    this.output_layers_templates.push(new Float32Array(totalSize));
    	                    this.parametric_layers.push(layerData.layer_name);
    	                } else if (layerData.layer_name === "maxPooling") {
    	                    newLayer = layerBuilder.maxPooling(layerData.poolSize, layerData.strides, layerData.padding);
    	                    newLayer.inputShape = layerData.inputShape;
    	                    newLayer.outputShape = layerData.outputShape;
    	                    const [H, W, D] = layerData.outputShape;
    	                    const totalSize = H * W * D;
    	                    this.output_layers_templates.push(new Float32Array(totalSize));
    	                }
    	                else if (layerData.layer_name === "EmbeddingLayer") {
    	                    const vocabSize = layerData.vocabSize;
    	                    const embeddingDim = layerData.embeddingDim;
    	                    const sequence_length = layerData.maxSequenceLength;
    	                    const outputSize = sequence_length * embeddingDim;
    	                    newLayer = layerBuilder.embeddingLayer(vocabSize, embeddingDim, sequence_length);
    	                    newLayer.inputShape = [];
    	                    newLayer.outputShape = [1, 1, outputSize];
    	                    newLayer.weightShape = [vocabSize, embeddingDim];
    	                    newLayer.outputSize = outputSize;
    	                    this.output_layers_templates.push(new Float32Array(outputSize));
    	                    this.parametric_layers.push(layerData.layer_name);
    	                }
    	                else if (layerData.layer_name === "recurrent_cell") {
    	                    const units = layerData.units;
    	                    const return_sequence = layerData.return_sequence || false;
    	                    const return_state = layerData.return_state || false;
    	                    newLayer = layerBuilder.recurrentCell(units, layerData.activation_function_name, return_sequence, return_state);
    	                    newLayer.weightShape = layerData.weightShape;
    	                    newLayer.inputShape = layerData.inputShape;
    	                    newLayer.outputShape = layerData.outputShape;
    	                    newLayer.maxSequenceLength = layerData.maxSequenceLength || 1;
    	                    this.output_layers_templates.push(new Float32Array(units));
    	                    this.parametric_layers.push(layerData.layer_name);
    	                }
    	                else {
    	                    throw new Error(`[ERROR] Unknown layer type '${layerData.layer_name}' found in model.`);
    	                }
    	                
    	                return newLayer;
    	            });

    	            console.log('[SUCCESS]------- Model loaded successfully');
    	        } catch (error) {
    	            console.error(error);
    	        }
    	    }

    	    get_task_type() {
    	        return this.task || "Task not specified";
    	    }

    	    /**
    	     * @method get_miscellaneous_data
    	     * @returns {Object} Saved miscellaneous data upon model saving
    	     */
    	    get_miscellaneous_data() {
    	        return this.miscellaneous;
    	    }

    	    /**
    	     *  @method predict()
    	        @param {Array} input - input data 
    	        @returns Array of predictions
    	        @throws Error when there's shape mismatch and no input data

    	     produces predictions based on the input data
    	    */
    	    async predict(input) {
    	        setGlobalParams(this.weights, this.biases, this.output_layers_templates);

    	        if (!this.weights || !this.biases || !this.output_layers_templates) {
    	            throw new Error("Parameters are missing");
    	        }

    	        try {
    	            if (!input) {
    	                throw new Error("\n[ERROR]-------No inputs")
    	            }

    	            for (let i = 0; i < input.length; i++) {
    	                if (input[i].length != (this.input_shape[0] * this.input_shape[1] * this.input_shape[2]) || input[i].length != this.input_size) {
    	                    this.isfailed = true;
    	                    console.log(`[ERROR]------- Input data must be the same shape set in the input layer\n- Use getTensorShape() or getInputSize()\n\nInput size/shape: ${input[i].length} || Expected: [${this.input_shape}] or ${this.input_size}\n`);
    	                    throw new Error(`Shape mismatch`);
    	                }

    	                input[i] = input[i] instanceof Float32Array ? input[i] : new Float32Array(input[i].flat(Infinity));
    	            }            

    	            let outputs = [];
    	            for (let sample_index = 0; sample_index < input.length; sample_index++) {
    	                let input_data = input[sample_index];

    	                const {predictions} = this.#Feedforward(input_data);
    	                outputs.push(predictions);
    	            }

    	            return outputs;
    	        }
    	        catch (error) {
    	            console.error(error);
    	        }
    	    }

    	    // ========= Private methods =======
    	    // forward propagation
    	    #Feedforward(input) {
    	        let current_input = input;
    	        let all_layer_outputs = [input];
    	        let zs = [];
    	        
    	        let outputTemplatePointer = 0;
    	        let pointer = 0;
    	        for (let layer_index = 0; layer_index < this.num_layers; layer_index++) {
    	            const current_layer = this.layers[layer_index];

    	            const { outputs, z_values, incrementor_value } = current_layer.feedforward(current_input, current_layer, pointer, outputTemplatePointer);
    	            pointer+=incrementor_value;
    	            outputTemplatePointer++;

    	            zs.push(z_values);
    	            current_input = outputs;
    	            all_layer_outputs.push(current_input);
    	        }

    	        return {
    	            predictions: current_input, 
    	            activations : all_layer_outputs,
    	            zs: zs
    	        };
    	    }
    	}

    	core = Runtime;
    	return core;
    }

    var tokenizer;
    var hasRequiredTokenizer;

    function requireTokenizer () {
    	if (hasRequiredTokenizer) return tokenizer;
    	hasRequiredTokenizer = 1;
    	const tokenize = (sentence) => {
    	    return sentence
    	        .toLowerCase()
    	        .match(/[a-z]+|\d+|[^\w\s]/g) || [];
    	};

    	const buildVocab = (sentences) => {
    	    const data = Array.isArray(sentences)
    	        ? sentences
    	        : [sentences];

    	    return [...new Set(data.flatMap(tokenize))];
    	};


    	const buildWord2Id = (vocab) => {
    	    const SPECIAL_TOKENS = ["<PAD>", "<UNK>"];
    	    return Object.fromEntries(
    	        [...SPECIAL_TOKENS, ...vocab].map((word, i) => [word, i])
    	    );
    	};

    	// Encode now just uses the pre-built word2id
    	const Encode = (sentence, word2id, max_length) => {

    	    let token_array = tokenize(sentence).map(word => word2id[word] ?? word2id["<UNK>"]);

    	    // sentences has different lengths, so we either pad (append 0s) or truncate the encoded input sentence
    	    
    	    // this block ensures to pad the input if it's less than the given max length
    	    if (token_array.length < max_length) {
    	        const needed_length = max_length - token_array.length;
    	        const pad = new Array(needed_length).fill(0);
    	        token_array = [...token_array, ...pad];
    	    }
    	    // this block is for truncating if the tokenize input is longer than the set maximumn length
    	    else if (token_array.length > max_length) {
    	        token_array.length = max_length;
    	    }

    	    return token_array;
    	};


    	tokenizer = {
    	    buildVocab,
    	    buildWord2Id,
    	    Encode,
    	    tokenize
    	};
    	return tokenizer;
    }

    var neurexruntime;
    var hasRequiredNeurexruntime;

    function requireNeurexruntime () {
    	if (hasRequiredNeurexruntime) return neurexruntime;
    	hasRequiredNeurexruntime = 1;
    	// const Runtime = require('./core/core.js');
    	// const { buildVocab, buildWord2Id, Encode, tokenize } = require('./preprocessor/tokenizer.js')

    	// exports.Runtime = Runtime;

    	const Runtime = requireCore();
    	const { buildVocab, buildWord2Id, Encode, tokenize } = requireTokenizer();


    	// under `NeurexRuntime` namespace, they can use the exported module like this:
    	// 
    	// const nrx = new NeurexRuntime.Runtime(); // this is the runtime class
    	// 
    	// const vocab = NeurexRuntime.buildVocab(extracted_text_data);
    	// const word2ID_output = NeurexRuntime.buildWord2Id(vocab);
    	// const encoded_data = NeurexRuntime.Encode("Hello World", word2ID_output, 20);
    	neurexruntime = {
    	  Runtime,
    	  buildVocab,
    	  buildWord2Id,
    	  Encode,
    	  tokenize
    	};
    	return neurexruntime;
    }

    var neurexruntimeExports = requireNeurexruntime();

    const { 
        Runtime, 
        buildVocab,
        buildWord2Id,
        Encode,
        tokenize
    } = neurexruntimeExports;

    exports.Encode = Encode;
    exports.Runtime = Runtime;
    exports.buildVocab = buildVocab;
    exports.buildWord2Id = buildWord2Id;
    exports.tokenize = tokenize;

    Object.defineProperty(exports, '__esModule', { value: true });

}));

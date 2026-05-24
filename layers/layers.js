/**
 * This is the Layers class, each layers (except inputShape()) has their own:
 * - initParams()
 * - determineInferenceType()
 * - feedforward()
 * - getOutputLayerDelta()
 * - backpropagate()
 * - computeWeightGrads()
 * - computeBiasGrads()
 * - scaleGrads()
 *
 * They'll be called during build time, feedforward, backpropagation, gradient accumulation and scaling gradients.
 * This is because Neurex follows a Plugin-style architecture where in modifications on the core engine (the core file) are minimal and the logic are exposed by these methods of the Layers class.
 * This allows the library to be extensible, flexible, and clean separation of concern without touching the core engine
 * Read here about Plugin-style architecture: https://medium.com/omarelgabrys-blog/plug-in-architecture-dec207291800
 */


const {
    getEmbeddings,
    returnEmbeddings,
    MatMul, 
    DeltaMatMul, 
    computeWeightGradientsForWeightsInConnectedLayer, 
    computeBiasGradsForConnected_Layer,
    applyPadding,
    Convolve,
    TransConvolve,
    Dilate_Input,
    ConvolveDelta,
    scaleGrads,
    element_wise_mul,
    element_wise_sub,
    computeBiasGradsForConv,
    MaxPool,
    MaxPoolDelta,
    ComputeGradientForKernels,
    scaleDiff
} = require('../core/bindings');

const {calculateTensorShape, getPaddingSizes, ifOneHotEndcoded, XavierInitialization, calculateTransConvOutputShape} = require('../utils/utils');
const activation = require('../core/bindings');
const { red, reset } = require('../color-code');


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
    inputShape(shapeConfig) {
        try {
            if (shapeConfig.features) {
                const features = shapeConfig.features;
                this.input_shape = null;
                return {
                    layer_name: "input_layer",
                    layer_size: features,
                    input_shape: null
                };
            } else if (shapeConfig.height && shapeConfig.width && shapeConfig.depth) {
                const { height, width, depth } = shapeConfig;

                return {
                    layer_name: "input_layer",
                    layer_size: height * width * depth,
                    input_shape: [height, width, depth]
                };
            } else {
                throw new Error(`[ERROR]------- Invalid input shape config`);
            }
        } catch (error) {
            console.error(error.message);
        }
    }

    /**
    * Creates an embedding layer for token encoding.
    *
    * @param {Number} vocabSize - The size of the vocabulary.
    * @param {Number} embeddingDim - The size of the dense vector used to represent each token.
    * @param {Number} maxSequenceLength - The length of the encoded token containing token IDs.
    * @returns {Object} - The embedding layer object configuration
    */
    emebeddingLayer(vocabSize, embeddingDim, maxSequenceLength) {
        if (vocabSize <= 0 || embeddingDim <= 0 || maxSequenceLength <= 0) throw new Error(`VocabSize or embeddingDim should not be a negative number or 0. vocabSize: ${vocabSize} | embeddingDim: ${embeddingDim} | maxSequenceLength: ${maxSequenceLength}`);

        return {
            layer_name:"EmbeddingLayer",
            vocabSize: vocabSize,
            embeddingDim: embeddingDim,
            maxSequenceLength: maxSequenceLength,
            initParams: (size, shape, layer_data) => {
                
                // Embedding layer can be added without input shape. So, we don't need to rely on the initial `size` and `shape` as it is just the default values from the constructor

                // the embedding layer will determine the input size and shape for the next layer
                const vocabSize = layer_data.vocabSize;
                const embeddingDim = layer_data.embeddingDim;

                const weightShape = [vocabSize, embeddingDim];
                const updatedShape = [1, 1,  maxSequenceLength * embeddingDim]; // this will be use for the next layer
                const updatedSize = maxSequenceLength * embeddingDim; // this will be use for the next layer

                // use Xavier Initialization: arg1 is the 'vocabSize' and; arg2 is the 'embeddingDim'
                const limit = XavierInitialization(vocabSize, embeddingDim);

                // creates a physical look up table to adjust where to put the <PAD> and <UNK>, this is row-major
                const lookUp = Array.from({length: vocabSize}, (_, i) => 
                    // the index 0 row is reserved for <PAD> and must be filled with 0s, same for <UNK> which is index as 1
                    i == 0 || i == 1 ? Array.from({length: embeddingDim}).fill(0) : Array.from({length: embeddingDim}, () => 
                        (Math.random() * 2 - 1) * limit
                    )
                )

                // flatten all to make it a 1D float32Array. Well index manually later on during feedforward and backprop (and gradient accumulation if it has to)
                const weights = new Float32Array(lookUp.flat(Infinity));

                // it has no biases, but since the core engine index weights and biases as well its corresponding zeroed initialized gradients together (meaning, if pointer index is 3, it will get the corresponding indexed weights and biases), 
                // we just initialize it with 0s equal to embedding size so that the weights and biases has no decrepancy when indexing and still match together
                const biases = new Float32Array(embeddingDim).fill(0);
                
                const weightGrads = new Float32Array(vocabSize * embeddingDim).fill(0);

                const biasGrads = new Float32Array(embeddingDim).fill(0);

                const output_template = new Float32Array(maxSequenceLength * embeddingDim);

                return {
                    updatedSize: updatedSize,
                    updatedShape: updatedShape,
                    weights: weights,
                    biases: biases,
                    weightGrads: weightGrads,
                    biasGrads: biasGrads,
                    outputTensors: output_template,
                    inputShape: [],
                    outputShape: updatedShape,
                    paramShape: weightShape,
                    isParametric: true,
                    overrides: {
                        // on core.js, the this.input_shape and this.input_size will be overwritten by these values
                        input_shape: [1, 1, maxSequenceLength], // this tells that the input vector going to the embedding layer is 1 * 1 * maxSequenceLength. The length of the input vector must match the max sequence length
                        input_size: maxSequenceLength
                    }
                }
            },
            determineInferenceType: () => {
                throw new Error('Embedding layer cannot be an output layer.');
                process.exit(1);
            },
            feedforward: (input, current_layer, pointer, outputTemplatePointer) => {
                const embeddingDim = current_layer.embeddingDim;

                const output = getEmbeddings(input, embeddingDim, pointer, outputTemplatePointer);
                return {
                    outputs: output, 
                    z_values: output,
                    incrementor_value: 1
                };
            },
            getOutputLayerDelta: () => {
                throw new Error('Embedding layer cannot be an output layer.');
                process.exit(1);
            },
            backpropagate: (next_delta, zs, layer_index, current_layer, allWeights, activations, nextLayer, pointer) => {
                
                let delta = next_delta;

                if (nextLayer.layer_name === "connected_layer") {
                    const [inputSize, outputSize] = nextLayer.weightShape;
                    delta = DeltaMatMul(delta, inputSize, outputSize, pointer);
                }

                return {
                    current_delta: delta,
                    incrementor_value: 1
                }
            },
            computeWeightGradients: (activation_outputs, delta, weightGrads, layer_data) => returnEmbeddings(activation_outputs, delta, weightGrads, layer_data.embeddingDim),
            computeBiasGradients: (biasGrads, delta, layer_data) => {
                // Embedding has no real biases — we used dummy zeros
                // Just return as-is, nothing to compute
                return biasGrads;
            },
            scaleGrads: (grads, batchSize, layer_data) => scaleGrads(grads, batchSize)
        }
    }

    /**
     * @method connectedLayer
     * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear)
     * @param {Number} layer_size specify the number of neuron for this layer.
     * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
     * @returns {Object}
     *
     * Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will
     * build connected layers or multilayer perceptron
     */
    connectedLayer(activation_function = 'relu', layer_size = 5) {
        try {

            if (!activation_function || !layer_size || layer_size <= 0) {
                throw new Error(`[ERROR]------- Layer Error | Activation function: ${activation_function} | layer size: ${layer_size}`);
            }

            let function_name = activation_function.toLowerCase();

            if (!activation[function_name] || !activation.derivatives[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                "layer_name":"connected_layer", 
                "activation_function":activation[function_name], 
                "derivative_activation_function":activation.derivatives[function_name],
                "layer_size":layer_size,
                initParams: (size, shape, layer_data) => {
                    const inputSize = size;
                    const outputSize = layer_data.layer_size;
                    const TotalWeightSize = outputSize * inputSize;
                    
                    const weights = new Float32Array(TotalWeightSize);
                    const weightGrads = new Float32Array(TotalWeightSize);
                    const biases = new Float32Array(outputSize);
                    const biasGrads = new Float32Array(outputSize);
                    const output_template = new Float32Array(outputSize);
                    
                    const limit = XavierInitialization(inputSize, outputSize);

                    for (let i = 0; i < TotalWeightSize; i++) {
                        weights[i] = (Math.random() * 2 - 1) * limit;
                    }
                    
                    for (let i = 0; i < outputSize; i++) {
                        biases[i] = (Math.random() * 2 - 1) * limit;
                    }

                    const weightShape = [inputSize, outputSize];
                    const updatedShape = [1, 1, outputSize]

                    return {
                        updatedSize: outputSize,
                        updatedShape: updatedShape,
                        weights: weights,
                        biases: biases,
                        weightGrads: weightGrads,
                        biasGrads: biasGrads,
                        outputTensors: output_template,
                        inputShape: [],
                        outputShape: updatedShape,
                        paramShape: weightShape,
                        isParametric: true,
                    }
                },
                determineInferenceType: (layerObject, lossFunc, trainY) => {
                    
                    let activation_function = layerObject.activation_function.name; // activation function
                    let layer_size = layerObject.layer_size; // layer size

                    /* do a loop to check if the trainY length are the same as output size if the loss is a categorical cross entropy and the activation function is softmax
                    * Example:
                    * output size: 3
                    * 
                    * The trainY should be:
                    * [
                    *    [0, 0, 1],
                    *    [1, 0, 0],
                    *    [0, 1, 0],
                    *    ....
                    * ]
                    */

                    if (lossFunc === "categorical_cross_entropy" && activation_function === "softmax") {
                        trainY.forEach(label => {
                            if (label.length != layer_size) throw new Error(`Output size must be the same number of classes. Number of classes: ${label.length} | Output layer size: ${layer_size}`);
                        });

                        // check also if the trainY are one hot encoded. Categorical Cross Entropy works wiht one-hot encoded labels
                        const isOneHotEncoded = ifOneHotEndcoded(trainY);
                        if (!isOneHotEncoded) throw new Error("Labels must be one hot encoded if the loss function is 'categorical_cross_entropy' and the activation function is `softmax`.");
                    }

                    if (lossFunc === "mae" || lossFunc === "mse") {
                        return "regression";
                    }

                    if (lossFunc === "binary_cross_entropy") {
                        return "binary_classification";
                    }

                    if (lossFunc === "categorical_cross_entropy" || lossFunc === "sparse_categorical_cross_entropy") {
                        return "multi_class_classification";
                    }

                    //  if none satisfies the conditions above, throw an error
                    throw new Error(`${red}[ERROR]------- Using ${lossFunc} having output size of ${layer_size} and an ${activation_function} function in the output layer is currently unavailable for this core's task.${reset}`);
                },
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => {

                    const [inputSize, outputSize] = current_layer.weightShape;
                    const z_values = MatMul(input, inputSize, outputSize, pointer, outputTemplatePointer);
                    const activation_function = activation[function_name];

                    let outputs = activation_function(z_values);
                    
                    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");
                    
                    return {
                        outputs, 
                        z_values,
                        incrementor_value: 1
                    };
                },
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
                    let dActivation = activation.derivatives[function_name];
                    let dOutputLayer = new Float32Array(preds.length); 

                    if (tasktype === "binary_classification" || (tasktype === "multi_class_classification" && lossFunc === "categorical_cross_entropy")) {
                        dOutputLayer = element_wise_sub(preds, actuals);
                    }
                    else if (tasktype === "multi_class_classification" && lossFunc === "sparse_categorical_cross_entropy") {
                        dOutputLayer.set(preds);
                        dOutputLayer[actuals[0]] -= 1;
                        
                    }
                    else if (tasktype === "regression") {
                        if (preds.length != actuals.length) {
                            throw new Error("Predictions array is not equal to actuals array");
                        }

                        const lastLayerZs = zs[zs.length - 1]; 
                        const dAct = dActivation(lastLayerZs); 

                        dOutputLayer = scaleDiff(preds, actuals, dAct);

                        if (dOutputLayer.some(v => Number.isNaN(v))) throw new Error("Delta of the output layer has NaNs"); 

                    }

                    return dOutputLayer;
                },
                backpropagate: (next_delta, zs, layer_index, current_layer, allWeights, activations, nextLayer, pointer) => {
                    const dActivation = activation.derivatives[function_name];
                    const [inputSize, outputSize] = nextLayer.weightShape;
                    const dAct = dActivation(zs[layer_index]);
                    const delta_res = DeltaMatMul(next_delta, inputSize, outputSize, pointer);
                    const current_delta = element_wise_mul(dAct, delta_res);

                    if (current_delta.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");            

                    return {
                        current_delta,
                        decrementor_value: 1
                    };
                },
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => computeWeightGradientsForWeightsInConnectedLayer(activation_outputs, deltas, weightGrads, layer_data.weightShape[0], layer_data.weightShape[1]),
                computeBiasGradients: (biasgrads, deltas, layer_data) => computeBiasGradsForConnected_Layer(biasgrads, deltas),
                scaleGrads: (grads, batchSize, layer_data) => scaleGrads(grads, batchSize)
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
                "layer_name":"convolutionalLayer",
                "activation_function":activation[function_name],
                "derivative_activation_function":activation.derivatives[function_name],
                "kernel_size":kernel_size,
                "filters":filters,
                "padding":padding.toLowerCase(),
                "strides":strides,
                initParams: (size, shape, layer_data) => {
                    const filters = layer_data.filters;
                    const [kH, kW] = layer_data.kernel_size;
                    const stride = layer_data.strides || 1;
                    const padding = layer_data.padding || "same";

                    const inputH = shape[0];
                    const inputW = shape[1];
                    const inputDepth = shape[2];

                    const inputShape = [inputH, inputW, inputDepth];

                    const TotalSize = filters * kH * kW * inputDepth;

                    let kernels = new Float32Array(TotalSize);
                    let kernelGrads = new Float32Array(TotalSize);
                    let biases = new Float32Array(filters);
                    let biasGrads = new Float32Array(filters);

                    const fanIn = kH * kW * inputDepth;
                    const fanOut = kH * kW * filters;
                    const limit = XavierInitialization(fanIn, fanOut);

                    for (let i = 0; i < TotalSize; i++) {
                        kernels[i] = (Math.random() * 2 - 1) * limit;
                    }

                    for (let i = 0; i < filters; i++) {
                        biases[i] = (Math.random() * 2 - 1) * limit;
                    }

                    // Calculate output shape
                    const { OutputHeight, OutputWidth, CalculatedTensorShape } = calculateTensorShape(inputH, inputW, kH, kW, filters, stride, padding);
                    const output_template = new Float32Array(CalculatedTensorShape)
                    // store output shape too
                    const outputShape = [OutputHeight, OutputWidth, filters];

                    const weightShape = [filters, kH, kW, inputDepth];
                    
                    return {
                        updatedSize: CalculatedTensorShape,
                        updatedShape: outputShape,
                        weights: kernels,
                        biases: biases,
                        weightGrads: kernelGrads,
                        biasGrads: biasGrads,
                        outputTensors: output_template,
                        inputShape: inputShape,
                        outputShape: outputShape,
                        paramShape: weightShape,
                        isParametric: true,
                    }
                },
                determineInferenceType: (layerObject, lossFunc, trainY) => {

                    if (lossFunc === "categorical_cross_entropy" && activation_function === "softmax") {
                        // check if the trainY are one hot encoded. Categorical Cross Entropy works wiht one-hot encoded labels
                        const isOneHotEncoded = ifOneHotEndcoded(trainY);
                        if (!isOneHotEncoded) throw new Error("Labels must be one hot encoded if the loss function is 'categorical_cross_entropy' and the activation function is `softmax`.");
                    }

                    if (activation_function === "linear" && (lossFunc === "mae" || lossFunc === "mse")) {
                        return "regression";
                    }

                    if ((activation_function === "sigmoid" || activation_function === "tanh") && (lossFunc === "binary_cross_entropy")) {
                        return "binary_classification";
                    }

                    if (activation_function === "softmax" && (lossFunc === "categorical_cross_entropy" || lossFunc === "sparse_categorical_cross_entropy")) {
                        return "multi_class_classification";
                    }

                    /**
                    * Convolution layers might have it's on way of determining task, I'll leave this as one of my TO DOs
                    */
                },
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => {
                    
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
                    const convolve_result = Convolve(data,current_layer.strides, OutputHeight, OutputWidth, f, kh, kw, kd, shape[0], shape[1], pointer, outputTemplatePointer);

                    if (convolve_result.some(Number.isNaN)) throw new Error('NaN detected on convolve result');

                    // 5. activate each depth input using the given activation function
                    const activation_function = activation[function_name];

                    const outputs = activation_function(convolve_result);

                    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");

                    return {
                        outputs: outputs,
                        z_values: convolve_result,
                        incrementor_value: 1
                    };
                },
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
                    /**
                    * Convolution layers has different process of getting delta of the output layer, so this is another TO DOs, but for now, throw an error 
                    */

                    throw new Error('Convolutional layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
                    process.exit(1);
                    // let dOutputLayer = new Float32Array(preds.length); 

                    // return dOutputLayer;
                },
                backpropagate: (next_delta, zs, layer_index, currentLayer, weights, activations, next_layer,pointer) => {
                    let Current_Z = zs[layer_index];
                    let dActivation = activation.derivatives[function_name];
                    let dL_dActivation;

                    if (next_layer.layer_name === "connected_layer") {
                        const [inputSize, outputSize] = next_layer.weightShape;
                        dL_dActivation = DeltaMatMul(next_delta, inputSize, outputSize, pointer);
                    } 
                    else if (next_layer.layer_name === "maxPooling") {
                        dL_dActivation = next_delta;
                    } 
                    else if (next_layer.layer_name === "convolutionalLayer") {
                        const [Fn, KHn, KWn, KCn] = next_layer.weightShape;
                        const [oHn, oWn, oDn] = next_layer.outputShape;
                        const [oHcurr, oWcurr] = currentLayer.outputShape;  // backward target shape
                        const stridesN = next_layer.strides;
                        const paddingN = next_layer.padding;

                        // dilate input
                        const {data: dilated, dilatedH, dilatedW} = Dilate_Input(next_delta, [oHn, oWn, oDn], stridesN);
                        
                        // const dilatedH = (oHn - 1) * stridesN + 1;
                        // const dilatedW = (oWn - 1) * stridesN + 1;

                        let pT, pB, pL, pR;
                        if (paddingN === "valid") {
                            // full conv: K-1 on every side
                            pT = pB = KHn - 1;
                            pL = pR = KWn - 1;
                        } else {
                            // "same": split K-1, then top up so the result is at least oHcurr/oWcurr
                            pT = Math.floor((KHn - 1) / 2); pB = (KHn - 1) - pT;
                            pL = Math.floor((KWn - 1) / 2); pR = (KWn - 1) - pL;

                            const needH = oHcurr + KHn - 1;          // ConvolveDelta needs Hp >= needH
                            const needW = oWcurr + KWn - 1;
                            const haveH = dilatedH + pT + pB;
                            const haveW = dilatedW + pL + pR;
                            if (haveH < needH) pB += (needH - haveH);
                            if (haveW < needW) pR += (needW - haveW);
                        }

                        // pass the REAL dilated dims, not oHn/oWn
                        const { data, shape } = applyPadding(dilated, dilatedH, dilatedW, oDn, pT, pB, pL, pR);

                        dL_dActivation = ConvolveDelta(data, shape, [Fn, KHn, KWn, KCn], oHcurr, oWcurr, pointer);
                    }
                    
                    const output = element_wise_mul(dActivation(Current_Z), dL_dActivation);
                    if (output.some(v => Number.isNaN(v))) throw new Error("Element-wise multiplication result has NaNs");

                    return {
                        current_delta: output,
                        decrementor_value: 1
                    };
                    
                    
                    
                },
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => {
                    const [filters, kH, kW, inDepth] = layer_data.weightShape
                    const [inH, inW] = layer_data.inputShape
                    const [outH, outW] = layer_data.outputShape


                    const output = ComputeGradientForKernels(activation_outputs,deltas,weightGrads,inH, inW, inDepth,outH, outW, filters,kH, kW);

                    if (output.some(Number.isNaN)) throw new Error(`Has NaNs after accumulation of kernel grads`);

                    return output;
                },
                computeBiasGradients: (biasgrads, deltas, layer_data) => {
                    const [filters] = layer_data.weightShape;
                    const [outH, outW] = layer_data.outputShape;

                    return computeBiasGradsForConv(biasgrads, deltas, outH, outW, filters);
                },
                scaleGrads: (grads, batchSize, layer_data) => scaleGrads(grads, batchSize)
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
                "layer_name":"maxPooling",
                "poolSize": poolSize,
                "padding": padding,
                "strides":strides,
                initParams: (size, shape, layer_data) => {
                    
                    // max pooling layer doesn't have parameters, so we just calculate what will be the output shape to be use for the next layer
                    const [inputH, inputW, inputD] = shape;
                    const [poolHeight, poolWidth] = layer_data.poolSize;
                    const strides = layer_data.strides || 1;
                    const padding = layer_data.padding || "same";

                    const inputShape = [inputH, inputW, inputD]; // set the input shape to be use in the feedforward() of maxPooling() layer

                    const weightShape = null;
                    const {OutputHeight, OutputWidth, CalculatedTensorShape} = calculateTensorShape(inputH, inputW, poolHeight, poolWidth, inputD, strides, padding); // we get the output shape to be use as input shape for the succeeding layers
                    const outputShape = [OutputHeight, OutputWidth, inputD]; // set the output shape
                    const output_template = new Float32Array(CalculatedTensorShape)

                    return {
                        updatedSize: CalculatedTensorShape,
                        updatedShape: outputShape,
                        weights: [],
                        biases: [],
                        weightGrads: [],
                        biasGrads: [],
                        outputTensors: output_template,
                        inputShape: inputShape,
                        outputShape: outputShape,
                        paramShape: weightShape,
                        isParametric: false,
                    }
                },
                determineInferenceType: (layerObject, lossFunc, trainY) => {
                    throw new Error('Max pooling layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
                    process.exit(1);
                },
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => {
                    const [inputh, inputw, inputd] = current_layer.inputShape;
                    const [outputh, outputw, outputd] = current_layer.outputShape;
                    const [poolHeight, poolWidth] = current_layer.poolSize;
                    const strides = current_layer.strides;
                
                    let {output, maxIndices} = MaxPool(input, [poolHeight, poolWidth], [inputh, inputw, inputd], [outputh, outputw, outputd], strides, outputTemplatePointer);

                    current_layer.maxIndices = maxIndices;

                    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");

                    return {
                        outputs:output,
                        z_values: output,
                        incrementor_value:0
                    }
                },
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
                    /**
                    * Max pooling has different process of getting delta of the output layer, so this is another TO DOs, but for now, throw an error 
                    */

                    throw new Error('Max pooling layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
                    process.exit(1);
                },
                backpropagate: (prev_delta, zs, layer_index, currentLayer, weights, activations, next_layer, pointer, outputTemplatePointer) => {
                    let next_delta = prev_delta;
                    const [inputH, inputW, inputD] = currentLayer.inputShape;
                    const [outputH, outputW, outputD] = currentLayer.outputShape;
                    const [poolHeight, poolWidth] = currentLayer.poolSize;
                    const strides = currentLayer.strides;
                    const padding = currentLayer.padding;

                    if (next_layer.layer_name === "connected_layer") {
                        const [inputSize, outputSize] = next_layer.weightShape;
                        next_delta = DeltaMatMul(prev_delta, inputSize, outputSize, pointer);
                    }

                    const indices = currentLayer.maxIndices;

                    const delta = MaxPoolDelta(new Float32Array(next_delta), indices, inputH, inputW, inputD);

                    return {
                        current_delta: delta,
                        decrementor_value:0
                    }
                },
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => {
                    // max pooling layer has no params like weights and biases, so no functions here :)
                },
                computeBiasGradients: (biasgrads, deltas, layer_data) => {
                    // max pooling layer has no params like weights and biases, so no functions here :)
                },
                scaleGrads: () => {
                    // max pooling layer has no params like weights and biases, so no functions here :)
                },
            }
        }
        catch (error) {
            console.error(error);
            process.exit(1);
        }
    }

    /**
     * 
     * @method transConvLayer
     * @param {Number} filters the number of filters for this convolutional layer. Produces the same number of output features
     * @param {Number} strides It determines how much the filter overlaps with the input as it slides across.
     * @param {Array<Number>} kernel_size the size of the kernel (or filter) that will slide and extracts input features
     * @param {String} activation_function the activation function to be use for this layer
     * @param {Number} padding adds N amount of padding on all sides. Default is 0
     * @param {Array<Number>} inputShape use to determine the shape of the input going to this layer, especially if the input comes from layers that works on 1D inputs (e.g. connected layers -> trans convolution where usual output shape of connected layers are [1, 1, outputSize]) 
     * @return {Object} transConv layer configs
     * @throws {Error} if any of the parameters are invalid.
     */
    transConvLayer(filters = 1, strides = 1, kernel_size = [3, 3], activation_function = 'relu', padding = 0, inputShape = [28, 28, 3]) {
        try {
            if (!filters || filters <= 0) throw new Error(`[ERROR]-------- Filters cannot be empty, less than or equal to 0. Filters: ${filters}`);
            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);
            if (!kernel_size || kernel_size.length == 0 || (kernel_size[0] <= 0 || kernel_size[1] <= 0)) throw new Error(`[ERROR]------- Kernels cannot be empty, nor it's height or width is less than or equal to 0. Kernel size: ${kernel_size}`);
            if (!activation_function || activation_function == undefined || activation_function == null || activation_function === "") throw new Error(`[ERROR]-------- activation_function cannot be empty, null or undefined.`);
            if (!padding || padding <= 0) throw new Error(`[ERROR]-------- Padding cannot be empty, null, undefined, or a negative value: ${padding}`);
            if (inputShape.length < 3) throw new Error(`[ERROR]------- [H, W, D] must be specified in the inputShape argumment: ${inputShape}`);
            if (inputShape[0] <=0 || inputShape[1] <= 0 || inputShape[2] <= 0) throw new Error(`[ERROR]------- Input shape values cannot be 0 or a negative number: ${inputShape}`);

            // check if the activation function is valid
            const function_name = activation_function.toLowerCase();

            if (!activation[function_name] || !activation.derivatives[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                layer_name: "transConv",
                activation_function: activation[function_name],
                derivative_activation_function: activation.derivatives[function_name],
                kernel_size: kernel_size,
                filters: filters,
                padding: padding,
                strides: strides,
                initParams: (size, shape, layer_data) => {
                    const [inputHeight, inputWidth, inputDepth] = inputShape;
                    const expectedSize = inputHeight * inputWidth * inputDepth;
                    if (size != expectedSize) throw new Error(`Previous layer size: ${size}. But transpose convolution having input shape ${inputShape} requires expected size: ${expectedSize}. `);

                    const filters = layer_data.filters
                    const [kernelHeight, kernelWidth] = layer_data.kernel_size;
                    const S = layer_data.strides;
                    const P = layer_data.padding;

                    const {OutputHeight, OutputWidth, OutputSize, OutputShape} = calculateTransConvOutputShape(inputHeight, inputWidth, kernelHeight, kernelWidth, filters, S, P);

                    const kernelShape = [filters, kernelHeight, kernelWidth, inputDepth];
                    const kernelSize = filters * kernelHeight * kernelWidth * inputDepth;

                    const weights = new Float32Array(kernelSize);
                    const biases = new Float32Array(filters);
                    const weightGrads = new Float32Array(kernelSize);
                    const biasGrads = new Float32Array(filters);
                    const output_tensor_template = new Float32Array(OutputSize);

                    const fanIn = kernelHeight * kernelWidth * inputDepth;
                    const fanOut = kernelHeight * kernelWidth * filters;
                    const limit = XavierInitialization(fanIn, fanOut);

                    for (let i = 0; i < kernelSize; i++) {
                        weights[i] = (Math.random() * 2 - 1) * limit;
                    }

                    for (let i = 0; i < filters; i++) {
                        biases[i] = (Math.random() * 2 - 1) * limit;
                    }

                    return {
                        updatedSize: OutputSize,
                        updatedShape: OutputShape,
                        weights: weights,
                        biases: biases,
                        weightGrads: weightGrads,
                        biasGrads: biasGrads,
                        outputTensors: output_tensor_template,
                        inputShape: inputShape,
                        outputShape: OutputShape,
                        paramShape: kernelShape,
                        isParametric: true,
                    }
                
                },
                determineInferenceType: (layerObject, lossFunc, trainY) => {
                    
                    let activation_function = layerObject.activation_function.name; // activation function
                    // let layer_size = layerObject.layer_size; // layer size

                    /* do a loop to check if the trainY length are the same as output size if the loss is a categorical cross entropy and the activation function is softmax
                    * Example:
                    * output size: 3
                    * 
                    * The trainY should be:
                    * [
                    *    [0, 0, 1],
                    *    [1, 0, 0],
                    *    [0, 1, 0],
                    *    ....
                    * ]
                    */

                    // if (lossFunc === "categorical_cross_entropy" && activation_function === "softmax") {
                    //     trainY.forEach(label => {
                    //         if (label.length != layer_size) throw new Error(`Output size must be the same number of classes. Number of classes: ${label.length} | Output layer size: ${layer_size}`);
                    //     });

                    //     // check also if the trainY are one hot encoded. Categorical Cross Entropy works wiht one-hot encoded labels
                    //     const isOneHotEncoded = ifOneHotEndcoded(trainY);
                    //     if (!isOneHotEncoded) throw new Error("Labels must be one hot encoded if the loss function is 'categorical_cross_entropy' and the activation function is `softmax`.");
                    // }

                    if (lossFunc === "mae" || lossFunc === "mse") {
                        return "regression";
                    }

                    // if (lossFunc === "binary_cross_entropy") {
                    //     return "binary_classification";
                    // }

                    // if (lossFunc === "categorical_cross_entropy" || lossFunc === "sparse_categorical_cross_entropy") {
                    //     return "multi_class_classification";
                    // }

                    //  if none satisfies the conditions above, throw an error
                    throw new Error(`${red}[ERROR]------- Using ${lossFunc} having output size of ${layer_size} and an ${activation_function} function in the output layer is currently unavailable for this core's task.${reset}`);
                },
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => {
                    
                    const strides = current_layer.strides;
                    const [inputHeight, inputWidth, inputDepth] = current_layer.inputShape;
                    const [OutputHeight, OutputWidth, OutputDepth] = current_layer.outputShape;
                    const [filters, kernelHeight, kernelWidth, kernelDepth] = current_layer.weightShape;

                    const {data, dilatedH, dilatedW} = Dilate_Input(input, [inputHeight, inputWidth, inputDepth], strides);

                    if (data.some(v => Number.isNaN(v))) throw new Error("[TRANS CONV ERROR] - dilation of input for transpose conv has NaNs");

                    const transConvRes = TransConvolve(data, strides, OutputHeight, OutputWidth, filters, kernelHeight, kernelWidth, inputDepth, dilatedH, dilatedW, pointer, outputTemplatePointer);

                    if (transConvRes.some(v => Number.isNaN(v))) throw new Error("[TRANS CONV ERROR] - Result of transpose conv has NaNs");

                    const activation_function = activation[function_name];

                    const outputs = activation_function(transConvRes);

                    if (outputs.some(v => Number.isNaN(v))) throw new Error("[TRANS CONV ERROR] - activation output array has Nans");

                    return {
                        outputs: outputs,
                        z_values: transConvRes,
                        incrementor_value:1
                    }

                },
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
                    let dActivation = activation.derivatives[function_name];
                    let dOutputLayer = new Float32Array(preds.length); 

                    if (tasktype === "binary_classification" || (tasktype === "multi_class_classification" && lossFunc === "categorical_cross_entropy")) {
                        dOutputLayer = element_wise_sub(preds, actuals);
                    }
                    else if (tasktype === "multi_class_classification" && lossFunc === "sparse_categorical_cross_entropy") {
                        dOutputLayer.set(preds);
                        dOutputLayer[actuals[0]] -= 1;
                        
                    }
                    else if (tasktype === "regression") {
                        if (preds.length != actuals.length) {
                            throw new Error("Predictions array is not equal to actuals array");
                        }

                        const lastLayerZs = zs[zs.length - 1]; 
                        const dAct = dActivation(lastLayerZs); 

                        dOutputLayer = scaleDiff(preds, actuals, dAct);

                        if (dOutputLayer.some(v => Number.isNaN(v))) throw new Error("Delta of the output layer has NaNs"); 

                    }

                    return dOutputLayer;
                },
                backpropagate: () => {

                },
                computeWeightGradients: () => {

                },
                computeBiasGradients: () => {

                },
                scaleGrads: () => {

                }

            }
        }
        catch (error) {
            console.error(error);
            process.exit(1);
        }
    }
}



module.exports = Layers;
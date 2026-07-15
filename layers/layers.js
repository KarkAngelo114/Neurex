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

const float32Ops = require('../core/bindings/float32Ops');
const {calculateTensorShape, getPaddingSizes, ifOneHotEndcoded, XavierInitialization, concatenateFloat32Array} = require('../utils/utils');
const activation = require('../core/bindings');
const { red, reset } = require('../color-code');

// import modular functions of different layers. We modularized it so that this file won't get bloated and contain 1k+ lines of code
const ann = require('../layers/layer_functions/connectedLayer');
const cnn = require('../layers/layer_functions/convolutionalLayer');


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
    embeddingLayer(vocabSize, embeddingDim, maxSequenceLength) {
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
                layer_name: "connected_layer", 
                activation_function: activation[function_name], 
                derivative_activation_function: activation.derivatives[function_name],
                layer_size: layer_size,
                initParams: (size, shape, layer_data) => ann.initParams(size, shape, layer_data),
                determineInferenceType: (layerObject, lossFunc, trainY) => ann.determineInferenceType(layerObject, lossFunc, trainY),
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => ann.feedforward(input, current_layer, pointer, outputTemplatePointer),
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => ann.getOutputLayerDelta(preds, actuals, zs, lossFunc, tasktype, layerObj),
                backpropagate: (delta, zs, layer_index, current_layer, nextLayer, pointer) => ann.backpropagate(delta, zs, layer_index, current_layer, nextLayer, pointer),
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
                layer_name: "convolutionalLayer",
                activation_function: activation[function_name],
                derivative_activation_function: activation.derivatives[function_name],
                kernel_size: kernel_size,
                filters: filters,
                padding: padding.toLowerCase(),
                strides: strides,
                initParams: (size, shape, layer_data) => cnn.initParams(size, shape, layer_data),
                determineInferenceType: () => cnn.determineInferenceType(),
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => cnn.feedforward(input, current_layer, pointer, outputTemplatePointer),
                getOutputLayerDelta: () => cnn.getOutputLayerDelta(),
                backpropagate: (delta, zs, layer_index, current_layer, nextLayer, pointer) => cnn.backpropagate(delta, zs, layer_index, current_layer, nextLayer, pointer),
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => {
                    const [filters, kH, kW, inDepth] = layer_data.weightShape
                    const [inH, inW] = layer_data.inputShape
                    const [outH, outW] = layer_data.outputShape


                    const output = ComputeGradientForKernels(
                        activation_outputs,
                        deltas,
                        weightGrads,
                        [inH, inW, inDepth],
                        [outH, outW, filters],
                        [kH, kW]);

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
                backpropagate: (prev_delta, zs, layer_index, currentLayer, next_layer, pointer, outputTemplatePointer) => {
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
                layer_size: units,
                initParams: (size, shape, layer_data) => {
                    const total_input_weights = size * units; // this is for the layer weights of this layer
                    const total_recurrent_weights = units * units; // this is for the recurrent weights of this layer
                    const totalBiases = units; // number of biases of this layer

                    const input_weights = new Float32Array(total_input_weights);
                    const recurrent_weights = new Float32Array(total_recurrent_weights);
                    const biases = new Float32Array(totalBiases);
                    const weightGrads = new Float32Array(total_input_weights + total_recurrent_weights); // combined the zeroed grads accumulator for input_weights and recurrent_weights
                    const biasGrads = new Float32Array(totalBiases);
                    const output_template = new Float32Array(units);

                    const limit1 = XavierInitialization(size, units);
                    const limit2 = XavierInitialization(units, units);
                    
                    for (let i = 0; i < total_input_weights; i++) {
                        input_weights[i] = (Math.random() * 2 - 1) * limit1;
                    }

                    for (let i = 0; i < total_recurrent_weights; i++) {
                        recurrent_weights[i] = (Math.random() * 2 - 1) * limit2;
                    }

                    for (let i = 0; i < totalBiases; i++) {
                        biases[i] = (Math.random() * 2 - 1) * limit1;
                    }

                    // concatenate input_weights and recurrent_weights. So the first size * units are the input_weights and the remaining units * units are the recurrent weights
                    // we concatenate both so that the optimizers can blindly consume 1D array (since it works on 1D arrays)
                    const concatenated_weights = concatenateFloat32Array([input_weights, recurrent_weights]);

                    // cache the number of input_weights so that we can just slice it later on.
                    layer_data.num_input_weights = total_input_weights;

                    const weightShape = [size, units];
                    const updatedShape = [1, 1, units];

                    return {
                        updatedSize: units,
                        updatedShape: updatedShape,
                        weights: concatenated_weights,
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
            }
        }
        catch (error) {
            console.error(error);
            process.exit(1);
        }
    }
}

module.exports = Layers;
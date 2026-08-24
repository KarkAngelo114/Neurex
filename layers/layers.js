/**
 * This is the Layers class, each layers (except inputShape()) has their own:
 * - initParams()
 * - determineInferenceType()
 * - feedforward()
 * - getOutputLayerDelta()
 * - projectDeltaBackward() 
 * - applyOwnDerivative()
 * - accumulateWeightGrads()
 * - accumulateBiasGrads()
 *
 */

const {
    computeWeightGradientsForWeightsInConnectedLayer, 
    computeBiasGradsForConnected_Layer,
} = require('../core/bindings/entry');

const activation = require('../core/bindings/entry');

// import modular functions of different layers. 
const inputConfig = require('./layer_functions/inputLayer');
const ann = require('./layer_functions/connectedLayer');
const cnn = require('./layer_functions/convolutionalLayer');
const maxpool = require('./layer_functions/maxPooling');
const embedding = require('./layer_functions/embeddingLayer');
const rnn = require('./layer_functions/recurrentCell');
const trans = require('./layer_functions/transConv');
const reshaper = require('./layer_functions/reshape');
const simple_attention = require("./layer_functions/simpleAttention")

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
     * @method reshape changes the dimensions (shape) of the data passing through it without changing the data values. This acts as the `input layer` to bridge data from layers that outputs 1D vector to be feed to convolutional layers which works on spatial grid-like data. 
     * @param targetShape specify the target shape for the data to be reshape. Default is `[28, 28, 3]`
     * @returns {Object} The reshape layer object configuration
    */
    reshape(targetShape = [28, 28, 3]) {
        if (targetShape.some(n => !n || n <= 0)) throw new Error(`[ERROR]------- Values should never be 0, null or a negative value.`);

        return {
            layer_name: 'Reshape',
            targetShape: targetShape,
            isParametric: false,
            initParams: (size, shape, layer_data) => reshaper.initParams(size, shape, layer_data),
            determineInferenceType: () => {  throw new Error('[ERROR]------- reshape cannot be an output layer') },
            feedforward: (input) => reshaper.feedforward(input),
            getOutputLayerDelta: () => {  throw new Error('[ERROR]------- reshape cannot be an output layer') },
            projectDeltaBackward: (delta) => delta,
            applyOwnDerivative: (delta) => delta,
            accumulateWeightGradients: () => {},
            accumulateBiasGradients: () => {},
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
            layer_name:"Embedding Layer",
            vocabSize: vocabSize,
            embeddingDim: embeddingDim,
            maxSequenceLength: maxSequenceLength,
            isParametric: true,
            initParams: (size, shape, layer_data) => embedding.initParams(size, shape, layer_data),
            determineInferenceType: () => embedding.determineInferenceType(),
            feedforward: (input, current_layer, pointer) => embedding.feedforward(input, current_layer, pointer),
            getOutputLayerDelta: () => embedding.getOutputLayerDelta(),
            projectDeltaBackward: (delta) => delta,
            applyOwnDerivative: (delta,) => delta,
            accumulateWeightGradients: (activation_outputs, delta, weightGrads, layer_data) => embedding.return_embeddings(activation_outputs, delta, weightGrads, layer_data),
            accumulateBiasGradients: (biasGrads) => biasGrads,
        }
    }

    /**
     * @method connectedLayer
     * @param {Number} layer_size specify the number of neuron for this layer. Default is `5`
     * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear, softmax). Default is `relu`.
     * @param {Boolean} useBias when set to `false`, the layer will not use bias and will skip bias initialization. Default value is `true`.
     * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
     * @returns {Object}
     *
     * Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will
     * build connected layers or multilayer perceptron
     */
    connectedLayer(layer_size = 5, activation_function = 'relu', useBias = true) {
        try {

            if (!activation_function || !layer_size || layer_size <= 0) {
                throw new Error(`[ERROR]------- Layer Error | Activation function: ${activation_function} | layer size: ${layer_size}`);
            }

            let function_name = activation_function.toLowerCase();

            if (!activation[function_name] || !activation.derivatives[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                layer_name: "Connected Layer", 
                activation_function: activation[function_name], 
                derivative_activation_function: activation.derivatives[function_name],
                layer_size: layer_size,
                isParametric: true,
                useBias: useBias,
                initParams: (size, shape, layer_data) => ann.initParams(size, shape, layer_data),
                determineInferenceType: (layerObject, lossFunc, trainY) => ann.determineInferenceType(layerObject, lossFunc, trainY),
                feedforward: (input, current_layer, pointer) => ann.feedforward(input, current_layer, pointer),
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => ann.getOutputLayerDelta(preds, actuals, zs, lossFunc, tasktype, layerObj),
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => ann.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                applyOwnDerivative: (delta, z, layer_data) => ann.applyOwnDerivative(delta, z, layer_data),
                accumulateWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => computeWeightGradientsForWeightsInConnectedLayer(activation_outputs, deltas, weightGrads, layer_data.weightShape[0], layer_data.weightShape[1]),
                accumulateBiasGradients: (biasgrads, deltas) => computeBiasGradsForConnected_Layer(biasgrads, deltas),
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
     * @param {Boolean} useBias when set to `false`, the layer will not use bias and will skip bias initialization. Default value is `true`.
     * @throws {Error} - if any of the parameters are invalid.
     * @returns {Object}
     *
     * Allows you to add convolutional layers in your model architecture in sequential building.
     */
    convolutionalLayer(filters = 1, strides = 1, kernel_size = [3, 3], activation_function = 'relu', padding = 'same', useBias = true) {
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
                layer_name: "Convolutional Layer",
                activation_function: activation[function_name],
                derivative_activation_function: activation.derivatives[function_name],
                kernel_size: kernel_size,
                filters: filters,
                padding: padding.toLowerCase(),
                strides: strides,
                isParametric: true,
                useBias: useBias,
                initParams: (size, shape, layer_data) => cnn.initParams(size, shape, layer_data),
                determineInferenceType: () => cnn.determineInferenceType(),
                feedforward: (input, current_layer, pointer) => cnn.feedforward(input, current_layer, pointer),
                getOutputLayerDelta: () => cnn.getOutputLayerDelta(),
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => cnn.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                applyOwnDerivative: (delta, z, layer_data) => cnn.applyOwnDerivative(delta, z, layer_data),
                accumulateWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => cnn.computeWeightGradients(activation_outputs, deltas, weightGrads, layer_data),
                accumulateBiasGradients: (biasgrads, deltas, layer_data) => cnn.computeBiasGradients(biasgrads, deltas, layer_data),
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
                layer_name: "Max Pooling",
                poolSize: poolSize,
                padding: padding,
                strides: strides,
                isParametric: false,
                initParams: (size, shape, layer_data) => maxpool.initParams(size, shape, layer_data),
                determineInferenceType: () => maxpool.determineInferenceType(),
                feedforward: (input, current_layer, pointer) => maxpool.feedforward(input, current_layer, pointer),
                getOutputLayerDelta: () => maxpool.getOutputLayerDelta(),
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => maxpool.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                applyOwnDerivative: (delta, z, layer_data) => maxpool.applyOwnDerivative(delta, z, layer_data),
                accumulateWeightGradients: () => {},
                accumulateBiasGradients: () => {},
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
     * @param {Boolean} useBias when set to `false`, the layer will not use bias and will skip bias initialization. Default value is `true`.
     */
    recurrentCell(units, activation_function = "tanh", return_sequence = false, useBias = true) {
        try {
            let function_name = activation_function.toLowerCase();

            if (!activation[function_name] || !activation.derivatives[function_name])  throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid.`);
            if (!units || units <= 0) throw new Error(`[ERROR]------- Units cannot be null, negative integer or a 0. | Units: ${units}`);

            return {
                layer_name: "Recurrent Cell", 
                activation_function: activation[function_name], 
                derivative_activation_function: activation.derivatives[function_name],
                units: units,
                return_sequence: return_sequence,
                isParametric: true,
                useBias: useBias,
                initParams: (size, shape, layer_data) => rnn.initParams(size, shape, layer_data),
                determineInferenceType: (layerObject, lossFunc, trainY) => rnn.determineInferenceType(layerObject, lossFunc, trainY),
                feedforward: (input, current_layer, pointer) => rnn.feedforward(input, current_layer, pointer),
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => rnn.getOutputLayerDelta(preds, actuals, zs, lossFunc, tasktype, layerObj),
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => rnn.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                applyOwnDerivative: (delta, z, layer_data) => rnn.applyOwnDerivative(delta, z, layer_data),
                accumulateWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => rnn.accumulateRecurrentWeightGrads(activation_outputs, deltas, weightGrads, layer_data),
                accumulateBiasGradients: (biasgrads, deltas, layer_data) => rnn.accumulateRecurrentBiasGrads(biasgrads, deltas, layer_data),
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
     * @param {String} padding adds N amount of padding on all sides. Default is 0
     * @param {Array<Number>} inputShape use to determine the shape of the input going to this layer, especially if the input comes from layers that works on 1D inputs (e.g. connected layers -> trans convolution where usual output shape of connected layers are [1, 1, outputSize])
     * @param {Boolean} useBias when set to `false`, the layer will not use bias and will skip bias initialization. Default value is `true`.
     * @return {Object} transConv layer configs
     * @throws {Error} if any of the parameters are invalid.
     */
    transConvLayer(filters = 1, strides = 1, kernel_size = [3, 3], activation_function = 'relu', padding = "Same", inputShape = [28, 28, 1], useBias = true) {
        try {
            if (!filters || filters <= 0) throw new Error(`[ERROR]-------- Filters cannot be empty, less than or equal to 0. Filters: ${filters}`);
            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);
            if (!kernel_size || kernel_size.length == 0 || (kernel_size[0] <= 0 || kernel_size[1] <= 0)) throw new Error(`[ERROR]------- Kernels cannot be empty, nor it's height or width is less than or equal to 0. Kernel size: ${kernel_size}`);
            if (!activation_function || activation_function == undefined || activation_function == null || activation_function === "") throw new Error(`[ERROR]-------- activation_function cannot be empty, null or undefined.`);
            if (!padding || padding == undefined || padding == null || padding === "") throw new Error(`[ERROR]-------- Padding cannot be empty, null or undefined.`);
            if (inputShape.some(num => !(num > 0))) throw new Error('[ERROR]------- Input shape values should not be null, undefined, 0 or a negative number')

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
                layer_name: "Trans Convolution",
                activation_function: activation[function_name],
                derivative_activation_function: activation.derivatives[function_name],
                kernel_size: kernel_size,
                filters: filters,
                padding: padding.toLowerCase(),
                strides: strides,
                inputShape: inputShape,
                isParametric: true,
                useBias: useBias,
                initParams: (size, shape, layer_data) => trans.initParams(size, shape, layer_data),
                determineInferenceType: (layerObject, lossFunc, trainY) => trans.determineInferenceType(layerObject, lossFunc, trainY),
                feedforward: (input, current_layer, pointer) => trans.feedforward(input, current_layer, pointer),
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => trans.getOutputLayerDelta(preds, actuals, zs, lossFunc, tasktype, layerObj),
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => trans.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                applyOwnDerivative: (delta, z, layer_data) => trans.applyOwnDerivative(delta, z, layer_data),
                accumulateWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => trans.accumulateKernelGrads(activation_outputs, deltas, weightGrads, layer_data),
                accumulateBiasGradients: (biasgrads, deltas, layer_data) => trans.accumulateBiasGradients(biasgrads, deltas, layer_data),
            }
        }
        catch (error) {
            console.error(error);
            process.exit(1);
        }
    }

    /**
     * @method `simpleAttention` layer is the implementation of an attention mechanism in its basic form.
     * @param {Boolean} useBias when set to `false`, the layer will not use bias and will skip bias initialization. Default value is `true`. 
     * @return {Object} simple attention layer configs
     */
    simpleAttention(useBias = true) {

        let function_name = "softmax";// default is softmax

        return {
            layer_name: "Simple Attention",
            useBias: useBias,
            activation_function: activation[function_name], 
            derivative_activation_function: activation.derivatives[function_name],
            isParametric: true,
            useBias: useBias,
            initParams: (size, shape, layer_data) => simple_attention.initParams(size, shape, layer_data),
            determineInferenceType: (layerObject, lossFunc, trainY) => simple_attention.determineInferenceType(layerObject, lossFunc, trainY),
            feedforward: (input, current_layer, pointer) => simple_attention.feedforward(input, current_layer, pointer),
            getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => simple_attention.getOutputLayerDelta(preds, actuals, zs, lossFunc, tasktype, layerObj),
            projectDeltaBackward: (delta, pointer, targetShape, layer_data) => simple_attention.projectDeltaBackward(delta, pointer, targetShape, layer_data),
            applyOwnDerivative: (delta, z, layer_data) => simple_attention.applyOwnDerivative(delta, z, layer_data),
            accumulateWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => simple_attention.accumulateKernelGrads(activation_outputs, deltas, weightGrads, layer_data),
            accumulateBiasGradients: (biasgrads, deltas, layer_data) => simple_attention.accumulateBiasGradients(biasgrads, deltas, layer_data),
        };
    }

}

module.exports = Layers;

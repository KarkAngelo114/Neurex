/**
 * This is the Layers class, each layers (except inputShape()) has their own:
 * - initParams()
 * - determineInferenceType()
 * - feedforward()
 * - getOutputLayerDelta()
 * - projectDeltaBackward()   ← replaces the old backpropagate(); called on the NEXT layer
 * - applyOwnDerivative()     ← replaces the old backpropagate(); called on the CURRENT layer
 * - computeWeightGrads()
 * - computeBiasGrads()
 * - scaleGrads()
 *
 * The two-method split eliminates all `if (nextLayer.layer_name === ...)` branching from
 * backprop. The core engine simply calls:
 *
 *   const dLda        = next_layer.projectDeltaBackward(delta, pointer, current_layer.outputShape, next_layer);
 *   current_delta     = current_layer.applyOwnDerivative(dLda, zs[layer_index], current_layer);
 *
 * JS dispatches to the right implementation automatically — same pattern as feedforward.
 * See core.js #backpropagation for the full driver loop.
 */

const {
    returnEmbeddings,
    computeWeightGradientsForWeightsInConnectedLayer, 
    computeBiasGradsForConnected_Layer,
    scaleGrads,
} = require('../core/bindings/entry');

const activation = require('../core/bindings/entry');
const { red, reset } = require('../color-code');

// import modular functions of different layers. 
const inputConfig = require('./layer_functions/inputLayer');
const ann = require('./layer_functions/connectedLayer');
const cnn = require('./layer_functions/convolutionalLayer');
const maxpool = require('./layer_functions/maxPooling');
const embedding = require('./layer_functions/embeddingLayer');
const rnn = require('./layer_functions/recurrentCell')


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
            initParams: (size, shape, layer_data) => embedding.initParams(size, shape, layer_data),
            determineInferenceType: () => embedding.determineInferenceType(),
            feedforward: (input, current_layer, pointer, outputTemplatePointer) => embedding.feedforward(input, current_layer, pointer, outputTemplatePointer),
            getOutputLayerDelta: () => embedding.getOutputLayerDelta(),
            projectDeltaBackward: (delta, pointer, targetShape, layer_data) => delta,
            applyOwnDerivative: (delta, z, layer_data) => delta,
            computeWeightGradients: (activation_outputs, delta, weightGrads, layer_data) => embedding.return_embeddings(activation_outputs, delta, weightGrads, layer_data),
            computeBiasGradients: (biasGrads, delta, layer_data) => biasGrads,
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
                // Step 1 of backprop: project delta backward through THIS layer's own weights.
                // Called as next_layer.projectDeltaBackward(...) from the core loop.
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => ann.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                // Step 2 of backprop: apply THIS layer's own activation derivative.
                // Called as current_layer.applyOwnDerivative(...) from the core loop.
                applyOwnDerivative: (delta, z, layer_data) => ann.applyOwnDerivative(delta, z, layer_data),
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
                // Step 1 of backprop: project delta backward through THIS conv layer's own kernels.
                // Called as next_layer.projectDeltaBackward(...) from the core loop.
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => cnn.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                // Step 2 of backprop: apply THIS layer's own activation derivative.
                // Called as current_layer.applyOwnDerivative(...) from the core loop.
                applyOwnDerivative: (delta, z, layer_data) => cnn.applyOwnDerivative(delta, z, layer_data),
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => cnn.computeWeightGradients(activation_outputs, deltas, weightGrads, layer_data),
                computeBiasGradients: (biasgrads, deltas, layer_data) => cnn.computeBiasGradients(biasgrads, deltas, layer_data),
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
                layer_name: "maxPooling",
                poolSize: poolSize,
                padding: padding,
                strides: strides,
                initParams: (size, shape, layer_data) => maxpool.initParams(size, shape, layer_data),
                determineInferenceType: () => maxpool.determineInferenceType(),
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => maxpool.feedforward(input, current_layer, pointer, outputTemplatePointer),
                getOutputLayerDelta: () => maxpool.getOutputLayerDelta(),
                // Step 1 of backprop: maxPooling has no weights so this is a pure passthrough.
                // Called as next_layer.projectDeltaBackward(...) from the core loop.
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => maxpool.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                // Step 2 of backprop: unpool using saved maxIndices.
                // Called as current_layer.applyOwnDerivative(...) from the core loop.
                applyOwnDerivative: (delta, z, layer_data) => maxpool.applyOwnDerivative(delta, z, layer_data),
                computeWeightGradients: () => {},
                computeBiasGradients: () => {},
                scaleGrads: () => {},
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
                initParams: (size, shape, layer_data) => rnn.initParams(size, shape, layer_data),
                determineInferenceType: (layerObject, lossFunc, trainY) => rnn.determineInferenceType(layerObject, lossFunc, trainY),
                feedforward: (input, current_layer, pointer, outputTemplatePointer) => rnn.feedforward(input, current_layer, pointer, outputTemplatePointer),
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => rnn.getOutputLayerDelta(preds, actuals, zs, lossFunc, tasktype, layerObj),
                projectDeltaBackward: (delta, pointer, targetShape, layer_data) => rnn.projectDeltaBackward(delta, pointer, targetShape, layer_data),
                applyOwnDerivative: (delta, z, layer_data) => rnn.applyOwnDerivative(delta, z, layer_data),
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => rnn.accumulateRecurrentWeightGrads(activation_outputs, deltas, weightGrads, layer_data),
                computeBiasGradients: (biasgrads, deltas, layer_data) => rnn.accumulateRecurrentBiasGrads(biasgrads, deltas),
                scaleGrads: (grads, batchSize, layer_data) => scaleGrads(grads, batchSize)
            }
        }
        catch (error) {
            console.error(error);
            process.exit(1);
        }
    }
}

module.exports = Layers;

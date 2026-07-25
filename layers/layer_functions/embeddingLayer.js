const { XavierInitialization } = require("../../utils/utils");
const activation = require('../../core/bindings');
const { getEmbeddings, DeltaMatMul, returnEmbeddings, recurrentTimeDelta } = require('../../core/bindings');


/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, outputTensors: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>, isParametric: Boolean}}
 */
const initParams = (size, shape, layer_data) => {
    // Embedding layer can be added without input shape. So, we don't need to rely on the initial `size` and `shape` as it is just the default values from the constructor

    // the embedding layer will determine the input size and shape for the next layer
    const vocabSize = layer_data.vocabSize;
    const embeddingDim = layer_data.embeddingDim;
    const maxSequenceLength = layer_data.maxSequenceLength;

    const weightShape = [vocabSize, embeddingDim];
    const updatedShape = [1, 1, embeddingDim, maxSequenceLength]; // this will be use for the next layer
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
}

/**
 * Determeines what is the task the model is trained on
 * @param {Object} layerObject layer configuration object
 * @param {String} lossFunc loss function is used for training
 * @param {Float32Array} trainY target labels 
 * @returns {string} task type
 */
const determineInferenceType = (layerObject, lossFunc, trainY) => {
    throw new Error('Embedding layer cannot be an output layer.');
    process.exit(1);
}

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
}

/**
 * 
 * @param {Float32Array} preds array of predicton outputs 
 * @param {Float32Array} actuals array of target labels 
 * @param {Array<Float32Array>} zs array of pre-activated values (zs)
 * @param {String} lossFunc loss function used in training
 * @param {String} tasktype task type the model is trained for 
 * @param {Object} layerObj layer config object of the last layer
 * @returns {Float32Array} the delta of the output layer
 */
const getOutputLayerDelta = (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
    throw new Error('Embedding layer cannot be an output layer.');
    process.exit(1);
}

/**
 * 
 * @param {Float32Array} activation_outputs outputs during feedforward
 * @param {Float32Array} delta outputs during backpropagation
 * @param {Float32Array} weightGrads zero initialize gradients for accumulation
 * @param {Object} layer_data layer object configuration
 * @returns {Float32Array}
 */
const return_embeddings = (activation_outputs, delta, weightGrads, layer_data) => {

    const output = returnEmbeddings(activation_outputs, delta, weightGrads, layer_data.embeddingDim);

    
    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs returning embeddings");

    return output;
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    return_embeddings
}
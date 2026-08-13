const { MaxPoolDelta, MaxPool } = require("../../core/bindings");
const { calculateTensorShape } = require("../../utils/utils");

/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>}}
 */
const initParams = (size, shape, layer_data) => {
    // max pooling layer doesn't have parameters, so we just calculate what will be the output shape to be use for the next layer
    const [inputH, inputW, inputD] = shape;
    const [poolHeight, poolWidth] = layer_data.poolSize;
    const strides = layer_data.strides || 1;
    const padding = layer_data.padding || "same";

    const inputShape = [inputH, inputW, inputD]; // set the input shape to be use in the feedforward() of maxPooling() layer

    const weightShape = null;
    const {OutputHeight, OutputWidth, CalculatedTensorShape} = calculateTensorShape(inputH, inputW, poolHeight, poolWidth, inputD, strides, padding); // we get the output shape to be use as input shape for the succeeding layers
    const outputShape = [OutputHeight, OutputWidth, inputD]; // set the output shape

    return {
        updatedSize: CalculatedTensorShape,
        updatedShape: outputShape,
        weights: [],
        biases: [],
        weightGrads: [],
        biasGrads: [],
        inputShape: inputShape,
        outputShape: outputShape,
        paramShape: weightShape,
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
    throw new Error('Max pooling layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
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
const feedforward = (input, current_layer, pointer) => {
    const [inputh, inputw, inputd] = current_layer.inputShape;
    const [outputh, outputw, outputd] = current_layer.outputShape;
    const [poolHeight, poolWidth] = current_layer.poolSize;
    const strides = current_layer.strides;
                
    let {output, maxIndices} = MaxPool(input, [poolHeight, poolWidth], [inputh, inputw, inputd], [outputh, outputw, outputd], strides);

    current_layer.maxIndices = maxIndices;

    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");

    return {
        outputs: output,
        z_values: output,
        incrementor_value:0
    }
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
    throw new Error('Max pooling layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
    process.exit(1);
}

/**
 * Projects the incoming delta backward through this layer.
 * MaxPooling has NO learnable weights, so the delta passes through unchanged —
 * the actual "undo" work (unpooling) is done in applyOwnDerivative below.
 * Called on the *next* layer (in feedforward direction) from the core backprop loop.
 *
 * @param {Float32Array} delta - incoming delta from the layer ahead (in backprop direction)
 * @param {Number} pointer - weight pointer (unused here; maxPooling is non-parametric)
 * @param {Array<Number>} targetShape - output shape of the receiving layer (unused here)
 * @param {Object} layer_data - this layer's own configuration (unused here)
 * @returns {Float32Array} the delta unchanged (passthrough)
 */
const projectDeltaBackward = (delta, pointer, targetShape, layer_data) => {
    // No weights to project through — pass the delta straight through.
    // The unpooling itself is this layer's own operation and belongs in applyOwnDerivative.
    return delta;
}

/**
 * Applies this max-pooling layer's own inverse operation (unpooling via saved max indices).
 * Called on the *current* layer from the core backprop loop.
 *
 * For max pooling there is no activation derivative in the traditional sense;
 * this step IS the full "undo my transform" operation.
 *
 * @param {Float32Array} delta - projected delta (output of next_layer.projectDeltaBackward)
 * @param {Float32Array} z - pre-activation values for this layer (same as outputs for maxPooling; unused here)
 * @param {Object} layer_data - this layer's own configuration (inputShape, maxIndices)
 * @returns {Float32Array} unpooled delta (same spatial size as this layer's input)
 */
const applyOwnDerivative = (delta, z, layer_data) => {
    const [inputH, inputW, inputD] = layer_data.inputShape;
    const indices = layer_data.maxIndices;
    const output = MaxPoolDelta(new Float32Array(delta), indices, inputH, inputW, inputD);
    if (output.some(v => Number.isNaN(v))) throw new Error("MaxPoolDelta result has NaNs in applyOwnDerivative (maxPooling)");
    return output;
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyOwnDerivative,
}

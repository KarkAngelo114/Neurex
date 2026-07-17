const { DeltaMatMul, MaxPoolDelta, MaxPool } = require("../../core/bindings");
const { calculateTensorShape } = require("../../utils/utils");

/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, outputTensors: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>, isParametric: Boolean}}
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
 * 
 * @param {Float32Array} delta incoming delta 
 * @param {Array<Float32Array>} zs an array containing Z values 
 * @param {Number} layer_index current layer index
 * @param {Object} current_layer current layer configuration 
 * @param {Object} nextLayer the configs of the next layer (in feedforward pass direction)
 * @param {Number} pointer pointer value to be used in fetching corresponding parameters
 * @returns {{ current_delta: Float32Array, decrementor_value: Number }}
 */
const backpropagate = (delta, zs, layer_index, current_layer, nextLayer, pointer) => {
    let next_delta = delta;
    const [inputH, inputW, inputD] = current_layer.inputShape;
    const [outputH, outputW, outputD] = current_layer.outputShape;
    const [poolHeight, poolWidth] = current_layer.poolSize;
    const strides = current_layer.strides;
    const padding = current_layer.padding;

    if (nextLayer.layer_name === "connected_layer") {
        const [inputSize, outputSize] = nextLayer.weightShape;
        next_delta = DeltaMatMul(delta, inputSize, outputSize, pointer);
        if (next_delta.some(v => Number.isNaN(v))) throw new Error("DeltaMatMul in MaxPool backpropagate result has NaNs");
    }

    const indices = current_layer.maxIndices;

    const output = MaxPoolDelta(new Float32Array(next_delta), indices, inputH, inputW, inputD);
    if (output.some(v => Number.isNaN(v))) throw new Error("MaxPoolDelta result has NaNs");

    return {
        current_delta: output,
        decrementor_value:0
    }
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    backpropagate
}
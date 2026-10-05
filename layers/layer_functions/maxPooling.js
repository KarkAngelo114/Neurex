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
    console.error('Max pooling layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
    process.exit(1);
}


const feedforward = (data) => {
    const layerData = data.layerData;
    const input = data.input;
    

    const inputShape= layerData.inputShape;
    const outputShape = layerData.outputShape;
    const poolsize= layerData.poolSize;
    const strides = layerData.strides;
    const layerID = layerData.layerID;
                
    let {output, maxIndices} = MaxPool(input, poolsize, inputShape, outputShape, strides);

    layerData.maxIndices = maxIndices;

    if (output.some(v => Number.isNaN(v))) {
        console.error("NaN detected after pooling operation in maxpooling during feedforward");
        throw new Error("ERR_NAN_DETECTED");
    }

    return {
        outputs: output,
        z_values: output,
        incrementor_value:0
    }
}

const getOutputLayerDelta = () => {
    console.error('Max pooling layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
    process.exit(1);
}

const applyOwnDerivative = (data) => {

    const layerData = data.layerData;
    const [inputH, inputW, inputD] = layerData.inputShape;
    const indices = layerData.maxIndices;
    const delta = data.delta;

    const layerID = layerData.layerID;

    const output = MaxPoolDelta(delta, indices, inputH, inputW, inputD);
    if (output.some(v => Number.isNaN(v))) {
        console.error("NaN detected after unpooling operation in maxpooling during delta projection");
        throw new Error("ERR_NAN_DETECTED");
    }
    
    return output;
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    applyOwnDerivative,
}

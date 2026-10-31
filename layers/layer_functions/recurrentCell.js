const { createTensorBuffer, concatenateFloat32Array, ifOneHotEndcoded } = require("../../utils");
const bindings = require('../../core/bindings');
const errFlag = require('../../err_flags');
const activation = require('../../core/bindings')
const { recurrentForward, recurrentBackward, element_wise_sub, cacheOutputLayerDelta } = bindings;

/**
 * Initializes parameters for this layer.
 * Incoming shape is [H, W, D, S] when the previous layer produces sequences,
 * otherwise [H, W, D]. Use layer.reshape() to bridge incompatible layers.
 * @param {Number} size number of neurons for this layer
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer configuration
 */
const initParams = (size, shape, layer_data) => {
    const { units, useBias } = layer_data;

    if (shape.length < 3) {
        console.error(`Incoming tensor shape is not compatible with ${layer_data.layer_name}. Connect with layer.reshape() first.`);
        console.log("Expecting [H, W, D, S], received [H, W, D] instead.");
        throw new Error(errFlag.ERR_TENSOR_SHAPE_INCOMPATIBLE);
    }

    const feature_size = shape[2];
    const seqLen = shape[3] || 1;
    const return_sequence = layer_data.return_sequence || false;

    const input_weight_count = feature_size * units;
    const recurrent_weight_count = units * units;

    const input_weights = createTensorBuffer([input_weight_count], { prefilledWith: "xavier", min: feature_size, max: units }).data;
    const recurrent_weights = createTensorBuffer([recurrent_weight_count], { prefilledWith: "xavier", min: units, max: units }).data;
    const biases = useBias
        ? createTensorBuffer([units], { prefilledWith: "xavier", min: feature_size, max: units }).data
        : createTensorBuffer([units], { prefilledWith: "zeroes" }).data;

    // Gradient buffers. recurrentBackward accumulates into these in place.
    const weightGrads = createTensorBuffer([input_weight_count + recurrent_weight_count], { prefilledWith: "zeroes" }).data;
    const biasGrads = createTensorBuffer([units], { prefilledWith: "zeroes" }).data;

    // Layout: [ W_input (feature_size x units) | W_recurrent (units x units) ]
    const concatenated_weights = concatenateFloat32Array([input_weights, recurrent_weights]);

    layer_data.seqLen = seqLen;

    // return_sequence: keep every timestep. Otherwise emit only the last hidden state.
    const updatedShape = return_sequence ? [1, 1, units, seqLen] : [1, 1, units, 1];
    const outputUnits = return_sequence ? units * seqLen : units;

    return {
        updatedSize: outputUnits,
        updatedShape,
        weights: concatenated_weights,
        biases,
        weightGrads,
        biasGrads,
        inputShape: [1, 1, size],
        outputShape: updatedShape,
        paramShape: [feature_size, units, units, units],
    };
};

/**
 * Determines the training task from the loss function.
 * @param {Object} layerObject layer configuration object
 * @param {String} lossFunc loss function used for training
 * @param {Array<Array<Number>>} trainY target labels
 * @returns {string} task type
 */
const determineInferenceType = (layerObject, lossFunc, trainY) => {
    const activation_function = layerObject.activation_function.name;
    const layer_size = layerObject.layer_size || layerObject.units;

    if (lossFunc === "categorical_cross_entropy" && activation_function === "softmax") {
        trainY.forEach(label => {
            if (label.length != layer_size) {
                throw new Error(`Output size must match number of classes. Classes: ${label.length} | Output layer size: ${layer_size}`);
            }
        });
        if (!ifOneHotEndcoded(trainY)) {
            throw new Error("Labels must be one-hot encoded for 'categorical_cross_entropy' with 'softmax'.");
        }
    }

    if (lossFunc === "mae" || lossFunc === "mse") return "regression";
    if (lossFunc === "binary_cross_entropy") return "binary_classification";
    if (lossFunc === "categorical_cross_entropy" || lossFunc === "sparse_categorical_cross_entropy") {
        return "multi_class_classification";
    }

    throw new Error(`[ERROR] Loss "${lossFunc}" with a ${activation_function} output of size ${layer_size} is unsupported.`);
};

/**
 * Forward pass over the whole sequence in one native call.
 */
const feedforward = (data) => {
    const { layerData, pointer, modelID } = data;
    const layerID = layerData.layerID;
    const seqLen = layerData.seqLen || 1;
    const activationFuncString = layerData.activation_function.name;

    const { activatedOutput, z_values } = recurrentForward(
        data.input,
        layerData.weightShape,
        seqLen,
        activationFuncString,
        layerData.return_sequence,
        pointer,
        modelID,
        layerID
    );

    if (activatedOutput.some(Number.isNaN)) {
        console.error('NaN detected in activated outputs during feedforward in Recurrent Cell');
        throw new Error(errFlag.NAN_ERR);
    }
    if (z_values.some(Number.isNaN)) {
        console.error('NaN detected in z values during feedforward in Recurrent Cell');
        throw new Error(errFlag.NAN_ERR);
    }

    layerData.cache = { 
        layer_output: activatedOutput 
    };

    return {
        outputs: activatedOutput,
        z_values: z_values,
        incrementor_value: 1
    };
};

/**
 * Computes dL/dz for the output layer and caches it.
 * The result is already in pre-activation space (z), so the caller must
 * flag it with deltaAlreadyIncludesActivationDerivative = true downstream.
 */
const getOutputLayerDelta = (data) => {
    const layerObj = data.layerData;
    const preds = data.predictions;
    const actuals = data.actuals;
    const modelID = data.modelID;
    const lossFunc = data.loss;
    const zs = data.zs;
    const storedOutput = layerObj.cache.layer_output;
    const layerID = layerObj.layerID;


    let dActivation = activation.derivatives[layerObj.activation_function.name];
    let dOutputLayer = new Float32Array(preds.length); 

    if (lossFunc === "categorical_cross_entropy" || lossFunc === "binary_cross_entropy") {
        dOutputLayer = element_wise_sub(preds, actuals, modelID, layerID);
    }
    else if (lossFunc === "sparse_categorical_cross_entropy") {
        dOutputLayer.set(preds);
        if (!dOutputLayer[actuals[0]]) {
            throw new Error(`Actual index value not exist in range. Actual target label: ${actuals[0]} | Output layer size: ${preds.length}`)
        }
        dOutputLayer[actuals[0]] -= 1;
                        
    }
    else {
        if (preds.length != actuals.length) {
            console.error(`[${red}ERROR${reset}] Predictions array is not equal to actuals array. Prediction size: ${preds.length} || Target data output size:${actuals.length}`);
            throw new Error(errFlag.SHAPE_ERR);
        }

        const lastLayerZs = zs[zs.length - 1]; 
        const dAct = dActivation(lastLayerZs, storedOutput, modelID, layerID); 

        dOutputLayer = scaleDiff(preds, actuals, dAct, modelID, layerID);

        if (dOutputLayer.some(v => Number.isNaN(v))) {
            console.error("Delta of the output layer has NaNs");
            throw new Error(errFlag.NAN_ERR);
        } 

    }

    cacheOutputLayerDelta(dOutputLayer, modelID, layerID);

    return dOutputLayer;
   
}

/**
 * Pass-through. The layer above hands us dL/dh; no work is needed here.
 */
const projectDeltaBackward = (data) => data.delta;

/**
 * @returns {{delta: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array}}
 */
const applyDerivativeAndComputeGradients = (data) => {
    const { layerData } = data;

    const { delta, weightGrads, biasGrads } = recurrentBackward(
        data.delta,                                   
        data.input,          
        data.z_value,
        data.weightGrads,
        data.biasGrads,
        layerData.weightShape,
        layerData.seqLen,
        layerData.activation_function.name,
        layerData.return_sequence,
        data.deltaAlreadyIncludesActivationDerivative === true,
        data.pointer,
        data.modelID,
        layerData.layerID
    );

    return { delta: delta, weightGrads: weightGrads, biasGrads: biasGrads };
};

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyDerivativeAndComputeGradients,
};
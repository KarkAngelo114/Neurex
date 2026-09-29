const { MatMul, element_wise_sub, element_wise_mul, scaleDiff, DeltaMatMul, computeWeightGradientsForWeightsInConnectedLayer, computeBiasGradsForConnected_Layer } = require("../../core/bindings");
const { ifOneHotEndcoded, createTensorBuffer } = require("../../utils/utils");
const activation = require('../../core/bindings');
const { red, reset } = require("../../color-code");

const initParams = (size, shape, layer_data) => {
    const inputSize = size;
    const outputSize = layer_data.layer_size;
    const TotalWeightSize = outputSize * inputSize;
    const useBias = layer_data.useBias;
    
    let weights = createTensorBuffer([TotalWeightSize], {prefilledWith:'xavier', min: inputSize, max: outputSize}).data;
    let weightGrads = createTensorBuffer([TotalWeightSize], {prefilledWith:'zeroes'}).data;
    let biases = createTensorBuffer([outputSize], {prefilledWith:'zeroes'}).data;
    let biasGrads = createTensorBuffer([outputSize], {prefilledWith:'zeroes'}).data;
    
    if (useBias) {
        biases = createTensorBuffer([outputSize], {prefilledWith:'xavier', min: inputSize, max: outputSize}).data;
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
        inputShape: [1, 1, size],
        outputShape: updatedShape,
        paramShape: weightShape,
    }
}

const determineInferenceType = (layerObject, lossFunc, trainY) => {
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
    throw new Error(`${red}[ERROR]------- Using ${lossFunc} having output size of ${layer_size} and an ${activation_function} function in the output layer is currently unavailable.${reset}`);
}

const feedforward = (data) => {
    const input = data.input;
    const layerData = data.layerData; // data.LayerData holds the metadata object of a layer during init params and build time
    const pointer = data.pointer;
    const modelID = data.modelID;

    const [inputSize, outputSize] = layerData.weightShape;
    const z_values = MatMul(input, inputSize, outputSize, pointer, modelID);

    const activation_function = activation[layerData.activation_function.name];
    let outputs = activation_function(z_values, pointer, modelID);

    if (outputs.some(v => Number.isNaN(v))) {
        console.error("NaN detected during feedforward in connected layer");
        throw new Error("ERR_NAN_DETECTED");
    }

    layerData.cache = { 
        layer_output: outputs 
    };

    return { 
        outputs: outputs, 
        z_values: z_values, 
        incrementor_value: 1 
    };
}

const getOutputLayerDelta = (preds, actuals, zs, lossFunc, tasktype, layerObj) => {

    let dActivation = activation.derivatives[layerObj.activation_function.name];
    let dOutputLayer = new Float32Array(preds.length); 

    if (lossFunc === "categorical_cross_entropy" || lossFunc === "binary_cross_entropy") {
        dOutputLayer = element_wise_sub(preds, actuals);
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
            throw new Error("[ERROR] Output data shape mismatch");
        }

        const lastLayerZs = zs[zs.length - 1]; 
        const dAct = dActivation(lastLayerZs); 

        dOutputLayer = scaleDiff(preds, actuals, dAct);

        if (dOutputLayer.some(v => Number.isNaN(v))) throw new Error("Delta of the output layer has NaNs"); 

    }

    return dOutputLayer;
}

const projectDeltaBackward = (data) => {
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const delta = data.delta;

    const [inputSize, outputSize] = layerData.weightShape;

    const result = DeltaMatMul(delta, inputSize, outputSize, pointer, modelID);

    if (result.some(v => Number.isNaN(v))) {
        console.error("NaN detected during delta projection in connected layer");
        throw new Error("ERR_NAN_DETECTED");
    };

    return result;
}

const applyOwnDerivative = (data) => {
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const delta = data.delta;
    const z = data.z_value;

    const dActivation = activation.derivatives[layerData.activation_function.name];
    const storedOutput = layerData.cache.layer_output;

    const dAct = dActivation(z, storedOutput, pointer, modelID);

    const result = element_wise_mul(dAct, delta, pointer, modelID);

    if (result.some(v => Number.isNaN(v))) {
        console.error("NaN detected during derivative application in connected layer");
        throw new Error("ERR_NAN_DETECTED");
    }

    return result;
}

const gradientAccumulation = (data) => {
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const weightGrads = data.weightGrads;
    const biasGrads = data.biasGrads;
    const activation_outputs = data.activation_outputs;
    const deltas = data.deltas;

    const [inputSize, outputSize] = layerData.weightShape;

    const accumulatedWeightGrads = computeWeightGradientsForWeightsInConnectedLayer(
        activation_outputs,
        deltas,
        weightGrads,
        inputSize,
        outputSize,
        pointer,
        modelID
    );

    if (accumulatedWeightGrads.some(v => Number.isNaN(v))) {
        console.error("NaN detected during weightGrads accumulation in connected layer");
        throw new Error("ERR_NAN_DETECTED");
    };

    const accumulatedBiasGrads = computeBiasGradsForConnected_Layer(
        biasGrads,
        deltas,
        pointer,
        modelID
    );

    if (accumulatedBiasGrads.some(v => Number.isNaN(v))) {
        console.error("NaN detected during biasGrads accumulation in connected layer");
        throw new Error("ERR_NAN_DETECTED");
    };

    return {
        accumulatedWeightGrads: accumulatedWeightGrads,
        accumulatedBiasGrads: accumulatedBiasGrads
    }
}


module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyOwnDerivative,
    gradientAccumulation
}

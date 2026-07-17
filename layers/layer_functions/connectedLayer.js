const { MatMul, element_wise_sub, element_wise_mul, scaleDiff, DeltaMatMul } = require("../../core/bindings");
const { XavierInitialization, ifOneHotEndcoded } = require("../../utils/utils");
const activation = require('../../core/bindings');
const { red, reset } = require("../../color-code");

/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, outputTensors: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>, isParametric: Boolean}}
 */
const initParams = (size, shape, layer_data) => {
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
}

/**
 * Determeines what is the task the model is trained on
 * @param {Object} layerObject layer configuration object
 * @param {String} lossFunc loss function is used for training
 * @param {Float32Array} trainY target labels 
 * @returns {string} task type
 */
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

/**
 * The feedforward logic of this layer
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (input, current_layer, pointer, outputTemplatePointer) => {
    const [inputSize, outputSize] = current_layer.weightShape; // weight shape [input, output]
    const z_values = MatMul(input, inputSize, outputSize, pointer, outputTemplatePointer); // perform the MatMul() operation

    const activation_function = activation[current_layer.activation_function.name]; // activation function
    let outputs = activation_function(z_values); // use the activation function       
    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");
                    
    return {
        outputs, 
        z_values,
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
    let dActivation = activation.derivatives[layerObj.activation_function.name];
    let dOutputLayer = new Float32Array(preds.length); 

    if (tasktype === "binary_classification" || (tasktype === "multi_class_classification" && lossFunc === "categorical_cross_entropy")) {
        dOutputLayer = element_wise_sub(preds, actuals);
    }
    else if (tasktype === "multi_class_classification" && lossFunc === "sparse_categorical_cross_entropy") {
        dOutputLayer.set(preds);
        dOutputLayer[actuals[0]] -= 1;
                        
    }
    else if (tasktype === "regression") {
        if (preds.length != actuals.length) throw new Error("Predictions array is not equal to actuals array");

        const lastLayerZs = zs[zs.length - 1]; 
        const dAct = dActivation(lastLayerZs); 

        dOutputLayer = scaleDiff(preds, actuals, dAct);

        if (dOutputLayer.some(v => Number.isNaN(v))) throw new Error("Delta of the output layer has NaNs"); 

    }

    return dOutputLayer;
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
    let current_delta;
    const dActivation = activation.derivatives[current_layer.activation_function.name];
    const dAct = dActivation(zs[layer_index]);

    let next_delta = delta;
    const [inputSize, outputSize] = nextLayer.weightShape;
    const delta_res = DeltaMatMul(next_delta, inputSize, outputSize, pointer);
                        
    current_delta = element_wise_mul(dAct, delta_res);

    if (current_delta.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");            

    return {
        current_delta,
        decrementor_value: 1
    };
}

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    backpropagate
}
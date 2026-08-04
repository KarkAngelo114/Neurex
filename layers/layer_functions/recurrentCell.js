const { XavierInitialization, concatenateFloat32Array, ifOneHotEndcoded } = require("../../utils");
const activation = require('../../core/bindings');
const { recurrentMatMul, element_wise_sub, scaleDiff, element_wise_mul, DeltaMatMul, recurrentTimeDelta, recurrentWeightGradsAccumulation, recurrentBiasGradsAccumulation } = require('../../core/bindings');
/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, outputTensors: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>}}
 */
const initParams = (size, shape, layer_data) => {
    const units = layer_data.units;
    
    // 1. Correctly extract the sequence length and feature size from the embedding layer's output shape
    // shape format from embedding is: [1, 1, embeddingDim, maxSequenceLength]
    const feature_size = shape[2] || size; 
    const spatialSteps = (shape[0] || 1) * (shape[1] || 1);
    let maxSequenceLength = layer_data.maxSequenceLength || shape[3] || (spatialSteps > 1 ? spatialSteps : 1);
    
    const return_sequence = layer_data.return_sequence || false;

    // 2. Compute total input weights using feature_size instead of size
    const total_input_weights = feature_size * units; 
    const total_recurrent_weights = units * units; 
    const totalBiases = units; 

    const input_weights = new Float32Array(total_input_weights);
    const recurrent_weights = new Float32Array(total_recurrent_weights);
    const biases = new Float32Array(totalBiases);
    const weightGrads = new Float32Array(total_input_weights + total_recurrent_weights); 
    const biasGrads = new Float32Array(totalBiases);

    const outputUnits = return_sequence ? (units * maxSequenceLength) : units;
    const output_template = new Float32Array(units); // output template per timesteps

    const limit1 = XavierInitialization(feature_size, units); // Use feature_size
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

    const concatenated_weights = concatenateFloat32Array([input_weights, recurrent_weights]);
    
    // 3. Keep track of the actual step feature size here
    const weightShape = [feature_size, units, units, units]; 

    layer_data.maxSequenceLength = return_sequence ? maxSequenceLength : 1;
    const updatedShape = [1, 1, outputUnits];

    return {
        updatedSize: outputUnits,
        updatedShape: updatedShape,
        weights: concatenated_weights,
        biases: biases,
        weightGrads: weightGrads,
        biasGrads: biasGrads,
        outputTensors: output_template,
        inputShape: [],
        outputShape: updatedShape,
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
    let activation_function = layerObject.activation_function.name; // activation function
    let layer_size = layerObject.layer_size || layerObject.units; // layer size

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
    throw new Error(`${red}[ERROR]------- Using ${lossFunc} having output unit size of ${layer_size} and an ${activation_function} function in the output layer is currently unavailable.${reset}`);
}

/**
 * The feedforward logic of this layer
 * @param {Float32Array} inputSequence input sequence data 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (inputSequence, current_layer, pointer, outputTemplatePointer) => {

    const units = current_layer.units;
    // Assume inputSequence is flat: [units * sequence_length]
    const sequence_length = current_layer.maxSequenceLength || 1; 
    const feature_size = current_layer.weightShape[0]; // [feature_size/size, units]

    // 1. Initialize a clean hidden state vector for the start of THIS sequence sample
    let current_hidden = new Float32Array(units).fill(0); 

    // 2. Arrays to store history for Backpropagation Through Time (BPTT)
    const all_z_values = [];
    const all_hidden_states = []; 

    // 3. Loop through each time step sequentially
    for (let t = 0; t < sequence_length; t++) {
        
        // Extract x_t for the current time step
        let offset = t * feature_size;
        const sequence_data = inputSequence.subarray(offset, offset + feature_size);

        // Compute z_t = (x_t * W_x) + (current_hidden * W_h) + bias
        // Pass current_hidden explicitly so it doesn't leak between global samples
        const z_t = recurrentMatMul(
            sequence_data, 
            current_hidden, 
            [current_layer.weightShape[0], current_layer.weightShape[1]], 
            [current_layer.weightShape[2], current_layer.weightShape[3]], 
            pointer, 
            outputTemplatePointer
        );

        if (z_t.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs on Recurrent layer (feedforward)");
        
        // Update hidden state for the next step
        current_hidden = activation[current_layer.activation_function.name](z_t);

        // Record history for backprop
        all_z_values.push(z_t);
        all_hidden_states.push(new Float32Array(current_hidden));
    }

    // cache recurrent layer cell feedforward data
    current_layer.cache = {
        hidden_states: all_hidden_states,
        recurrentZs: all_z_values
    }

    let final_output;
    if (current_layer.return_sequence) {
        // Concatenate all hidden states into one big flat array if return_sequence is true
        final_output = concatenateFloat32Array(all_hidden_states);
    } else {
        // Just return the very last hidden state vector
        final_output = all_hidden_states[sequence_length - 1];
    }

    return {
        outputs: final_output, 
        z_values: all_z_values, // Pass the array of z_values back to Neurex
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
        if (!dOutputLayer[actuals[0]]) {
            throw new Error(`Actual index value not exist in range. Actual target label: ${actuals[0]} | Output layer size: ${preds.length}`)
        }
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


const projectDeltaBackward = (delta, pointer, targetShape, layer_data) => {
    const sequenceLength = layer_data.maxSequenceLength;
    const units = layer_data.units;
    const featureSize = layer_data.weightShape[0];
    const recurrentZs = layer_data.cache.recurrentZs;
    const dActivation = activation.derivatives[layer_data.activation_function.name];

    const inputDeltas = new Float32Array(sequenceLength * featureSize);
    const deltaTs = new Array(sequenceLength);
    let dNextTime = new Float32Array(units).fill(0);

    for (let t = sequenceLength - 1; t >= 0; t--) {
        let dUpper = new Float32Array(units);

        if (layer_data.return_sequence) {
            dUpper.set(delta.subarray(t * units, (t + 1) * units));
        } 
        else if (t === sequenceLength - 1) {
            dUpper.set(delta);
        }

        let dTotal = new Float32Array(units);
        for (let i = 0; i < units; i++) dTotal[i] = dUpper[i] + dNextTime[i];

        const delta_t = element_wise_mul(dTotal, dActivation(recurrentZs[t]));
        if (delta_t.some(v => Number.isNaN(v))) throw new Error("delta_t has NaNs in recurrentCell.projectDeltaBackward");

        deltaTs[t] = delta_t;

        dNextTime = recurrentTimeDelta(delta_t, [featureSize, units], [units, units], pointer);
        const dInput_t = DeltaMatMul(delta_t, featureSize, units, pointer);
        inputDeltas.set(dInput_t, t * featureSize);
    }

    layer_data.cache.deltaTs = deltaTs;
    return inputDeltas;
};


const applyOwnDerivative = (delta, z, layer_data) => delta;

const accumulateRecurrentWeightGrads = (activation_outputs, deltas, weightGrads, layer_data) => {
    const weightShape = layer_data.weightShape;
    const sequenceLength = layer_data.maxSequenceLength;
    const hiddenStates = layer_data.cache.hidden_states;
    const deltaTs = layer_data.cache.deltaTs;

    if (!deltaTs) throw new Error("recurrentCell: projectDeltaBackward must run before computeWeightGradients — missing cached per-timestep deltas");

    const output = recurrentWeightGradsAccumulation(activation_outputs, deltas, hiddenStates, deltaTs, weightGrads, weightShape, sequenceLength);

    if (output.some(v => Number.isNaN(v))) throw new Error("recurrentCell weight grads have NaNs");

    return output;
};

const accumulateRecurrentBiasGrads = (biasgrads, deltas, layer_data) => {
    const units = layer_data.units;
    const sequenceLength = layer_data.maxSequenceLength;
    const deltaTs = layer_data.cache.deltaTs;

    if (!deltaTs) throw new Error("recurrentCell: projectDeltaBackward must run before computeBiasGradients");

    const output = recurrentBiasGradsAccumulation(biasgrads, deltaTs, sequenceLength, units);

    if (output.some(v => Number.isNaN(v))) throw new Error("recurrentCell bias grads have NaNs");

    return output;
};

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyOwnDerivative,
    accumulateRecurrentWeightGrads,
    accumulateRecurrentBiasGrads
}
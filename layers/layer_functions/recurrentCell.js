const { XavierInitialization, concatenateFloat32Array, ifOneHotEndcoded } = require("../../utils");
const activation = require('../../core/bindings');
const { recurrentMatMul } = require('../../core/bindings');
/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, outputTensors: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>, isParametric: Boolean}}
 */
const initParams = (size, shape, layer_data) => {
    const units = layer_data.units;
    const maxSequenceLength = layer_data.maxSequenceLength || 1;
    const return_sequence = layer_data.return_sequence || false;

    const total_input_weights = size * units; 
    const total_recurrent_weights = units * units; 
    const totalBiases = units; 

    const input_weights = new Float32Array(total_input_weights);
    const recurrent_weights = new Float32Array(total_recurrent_weights);
    const biases = new Float32Array(totalBiases);
    const weightGrads = new Float32Array(total_input_weights + total_recurrent_weights); 
    const biasGrads = new Float32Array(totalBiases);

    // If returning the full sequence, the output size is multiplied by the steps
    const outputUnits = return_sequence ? (units * maxSequenceLength) : units;
    
    // Create the output template with the corrected size
    const output_template = new Float32Array(outputUnits);

    const limit1 = XavierInitialization(size, units);
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
    const weightShape = [size, units, units, units]; 

    // Update the shape representation to reflect the sequence dimensionality
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
 * @param {Float32Array} input input features 
 * @param {Object} current_layer current layer object coonfiguration
 * @param {Number} pointer a pointer to be used for getting the corresponding weights and biases
 * @param {Number} outputTemplatePointer a pointer to be used for getting the corresponding output tensor template
 * @returns {{ outputs: Float32Array, z_values: Float32Array, incrementor_value: Number }}
 */
const feedforward = (inputSequence, current_layer, pointer, outputTemplatePointer) => {
    const units = current_layer.units;
    // Assume inputSequence is flat: [sequence_length * feature_size]
    // You'll need to know the sequence length (e.g., passed from layer configuration)
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
        const x_t = inputSequence.subarray(t * feature_size, (t + 1) * feature_size);

        // Compute z_t = (x_t * W_x) + (current_hidden * W_h) + bias
        // Pass current_hidden explicitly so it doesn't leak between global samples
        const z_t = recurrentMatMul(
            x_t, 
            current_hidden, 
            [current_layer.weightShape[0], current_layer.weightShape[1]], 
            [current_layer.weightShape[2], current_layer.weightShape[3]], 
            pointer, 
            outputTemplatePointer
        ); 
        
        // Update hidden state for the next step
        current_hidden = activation[current_layer.activation_function.name](z_t);

        // Record history for backprop
        all_z_values.push(z_t);
        all_hidden_states.push(new Float32Array(current_hidden));
    }

    // 4. Format the output based on your return_sequence flag
    let final_output;
    if (current_layer.return_sequence) {
        // Concatenate all hidden states into one big flat array
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

module.exports = {
    initParams,
    determineInferenceType,
    feedforward
}
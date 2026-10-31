const { createTensorBuffer, concatenateFloat32Array, ifOneHotEndcoded } = require("../../utils");
const activation = require('../../core/bindings');
const errFlag = require('../../err_flags');
const { recurrentForward, recurrentBackward, element_wise_sub, cacheOutputLayerDelta } = require('../../core/bindings');
/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>}}
 */
const initParams = (size, shape, layer_data) => {
    const units = layer_data.units;
    const useBias = layer_data.useBias;
    // incoming shape is [H, W, D, S] if before the recurrent cell is another layer that works on sequential data, otherwise it's [H, W, D]
    // To properly connect layers that works differently, use reshape()

    if (shape.length < 3) {
        console.error(`Incoming tensor shape is not compatible to this layer type ${layer_data.layer_name} as this layer has different tensor shape representation of incoming tensor data. Ensure you connect with layer.reshape() first before connecting to this layer type`);
        console.log("Expecting [H, W, D, S], received [H, W, D] instead.");
        throw new Error(errFlag.ERR_TENSOR_SHAPE_INCOMPATIBLE);
    }


    const feature_size = shape[2];
    const seqLen = shape[3];
    
    const return_sequence = layer_data.return_sequence || false;

    // 2. Compute total input weights using feature_size instead of size
    const total_input_weights = feature_size * units; 
    const total_recurrent_weights = units * units; 
    const totalBiases = units; 

    const input_weights = createTensorBuffer([total_input_weights],{ prefilledWith: "xavier", min: feature_size, max: units }).data;
    const recurrent_weights = createTensorBuffer([total_recurrent_weights], { prefilledWith: "xavier", min: units, max: units }).data;
    const biases = useBias ? createTensorBuffer([totalBiases], { prefilledWith: "xavier", min: feature_size, max: units }).data : createTensorBuffer([totalBiases], { prefilledWith: "zeroes" }).data;
    const weightGrads = createTensorBuffer([total_input_weights + total_recurrent_weights], { prefilledWith: "zeroes" }).data;
    const biasGrads = createTensorBuffer([totalBiases], { prefilledWith: "zeroes" }).data;
    

    const concatenated_weights = concatenateFloat32Array([input_weights, recurrent_weights]);
    
    // 3. Keep track of the actual step feature size here
    const weightShape = [feature_size, units, units, units]; 

    layer_data.seqLen = return_sequence ? seqLen : 1;
    const updatedShape = return_sequence ? 
        [1, 1, units, seqLen] : // preserve the sequence length
        [1, 1, units, 1]; // return only the output unit size, sequence length will become 1 and if the next layer works on sequential data, that layer will process this layer's output as one vector for timestep

    const outputUnits = return_sequence ? (units * seqLen) : units;

    return {
        updatedSize: outputUnits,
        updatedShape: updatedShape,
        weights: concatenated_weights,
        biases: biases,
        weightGrads: weightGrads,
        biasGrads: biasGrads,
        inputShape: [1, 1, size],
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

const feedforward = (data) => {

    const inputSequence = data.input;
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;
    const seqLen = layerData.seqLen || 1; 
    const weightShape = layerData.weightShape;
    const activationFuncString = layerData.activation_function.name;
    const returnSequence = layerData.return_sequence;

    const {activatedOutput, z_values} = recurrentForward(inputSequence, weightShape, seqLen, activationFuncString, returnSequence, pointer, modelID, layerID);
    
    if (activatedOutput.some(v => Number.isNaN(v))) {
        console.error('NaN detected in activated outputs during feedfoward in Recurrent Cell');
        throw new Error(errFlag.NAN_ERR);
    }

    if (z_values.some(v => Number.isNaN(v))) {
        console.error('NaN detected in z values during feedfoward in Recurrent Cell');
        throw new Error(errFlag.NAN_ERR);
    }

    return {
        outputs: activatedOutput, // used in recurrentBackward for gradient caculation. `activatedOutput` are the activated hidden states per timestep. it just uses the name `activatedOutput` as naming convention
        z_values: z_values, // used in BPTT. These are the z per timesteps (pre-activated zs)
        incrementor_value: 1
    };
}

const getOutputLayerDelta = (data) => {
    const layerObj = data.layerData;
    const preds = data.predictions;
    const actuals = data.actuals;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const lossFunc = data.loss;
    const zs = data.zs;
    const storedOutput = layerObj.cache.storedOutput;
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
            throw new Error("[ERROR] Output data shape mismatch");
        }

        const lastLayerZs = zs[zs.length - 1]; 
        const dAct = dActivation(lastLayerZs, storedOutput, modelID, layerID); 

        dOutputLayer = scaleDiff(preds, actuals, dAct, modelID, layerID);

        if (dOutputLayer.some(v => Number.isNaN(v))) throw new Error("Delta of the output layer has NaNs"); 

    }

    cacheOutputLayerDelta(dOutputLayer, modelID, layerID);

    return dOutputLayer;
   
}


const projectDeltaBackward = (data) => {

    return data.delta; // just a pass through. No operations here

};

const applyDerivativeAndComputeGradients = (data) => {
    
    const incoming_delta = data.delta; // incoming delta
    const hiddenStates = data.activatedOutput; // activated outputs are the hidden states per timesteps
    const z_per_timeSteps = data.z_value; // z_values are the z_values per timesteps
    const weight_grads = data.weightGrads; // weightGrads: [inputWeights..., recurrentWeights]
    const bias_grads = data.biasGrads; // size equal to units
    const pointer = data.pointer;
    const modelID = data.modelID;

    const layerData = data.layerData;
    const seqLen = layerData.seqLen;
    const weightShape = layerData.wweightShape;
    const layerID = layerData.layerID;

    const {delta, weightGrads, biasGrads} = recurrentBackward();

    return {
        weightGrads: [],
        biasGrasds: [],
        delta: []
    }
}


// const applyOwnDerivative = (data) => {
//     const layer_data = data.layerData;
//     const delta = data.delta;
//     const pointer = data.pointer;
//     const modelID = data.modelID;
//     const layerID = layer_data.layerID;

    
//     const units = layer_data.units;
//     const featureSize = layer_data.weightShape[0];
//     const hiddenStates = layer_data.cache.hidden_states; // Array of Float32Array
//     const recurrentZs = layer_data.cache.recurrentZs; // Array of Float32Array
//     const dActivation = activation.derivatives[layer_data.activation_function.name];

//     const inputDeltas = new Float32Array(sequenceLength * featureSize);
//     const deltaTs = new Array(sequenceLength);
//     let dNextTime = new Float32Array(units).fill(0);

//     for (let t = sequenceLength - 1; t >= 0; t--) {
//         let dUpper = new Float32Array(units);

//         if (layer_data.return_sequence) {
//             dUpper.set(delta.subarray(t * units, (t + 1) * units));
//         } else if (t === sequenceLength - 1) {
//             dUpper.set(delta);
//         }

//         let dTotal = new Float32Array(units);
//         for (let i = 0; i < units; i++) {
//             dTotal[i] = dUpper[i] + dNextTime[i];
//         }

//         // pass both recurrentZs[t] and hiddenStates[t]
//         // the recurrenZs are array of float32array that are not yet passed through an activation function. While the hidden states are the activated outputs
//         const dAct = dActivation(recurrentZs[t], hiddenStates[t], modelID, layerID);
        
//         // For element-wise activations, dAct is dL/dz. 
//         // For Softmax, dsoftmax(a_t, dTotal) computes the Jacobian-vector product J^T * dTotal directly.
//         const delta_t = (layer_data.activation_function.name === "softmax") 
//             ? dAct // dsoftmax already calculates the vector product if passed (a, delta)
//             : element_wise_mul(dTotal, dAct, modelID, layerID);

//         if (delta_t.some(v => Number.isNaN(v))) throw new Error("delta_t has NaNs in recurrent cell projectDeltaBackward");

//         deltaTs[t] = delta_t;
//         dNextTime = recurrentTimeDelta(delta_t, [featureSize, units], [units, units], pointer, modelID);
//         const dInput_t = DeltaMatMul(delta_t, featureSize, units, pointer, modelID, layerID);
//         inputDeltas.set(dInput_t, t * featureSize);
//     }

//     layer_data.cache.deltaTs = deltaTs;

//     return inputDeltas;
// };

// const gradientAccumulation = (data) => {
//     const layer_data = data.layerData;
//     const activation_outputs = data.activation_outputs;
//     const deltas = data.deltas;
//     const pointer = data.pointer;
//     const modelID = data.modelID;
//     const weightGrads = data.weightGrads;
//     const biasGrads = data.biasGrads;
//     const layerID = layer_data.layerID;

//     const units = layer_data.units;
//     const weightShape = layer_data.weightShape;
//     const sequenceLength = layer_data.maxSequenceLength;
//     const hiddenStates = layer_data.cache.hidden_states;
//     const deltaTs = layer_data.cache.deltaTs;

//     if (!deltaTs) throw new Error("recurrentCell: projectDeltaBackward must run before computeWeightGradients — missing cached per-timestep deltas");

//     const recurrentWeightGrads = recurrentWeightGradsAccumulation(activation_outputs, deltas, hiddenStates, deltaTs, weightGrads, weightShape, sequenceLength);
//     if (recurrentWeightGrads.some(v => Number.isNaN(v))) throw new Error("recurrentCell weight grads have NaNs");

//     const recurrentBiasGrads = recurrentBiasGradsAccumulation(biasGrads, deltaTs, sequenceLength, units);
//     if (recurrentBiasGrads.some(v => Number.isNaN(v))) throw new Error("recurrentCell bias grads have NaNs");

//     return {
//         accumulatedWeightGrads: recurrentWeightGrads,
//         accumulatedBiasGrads: recurrentBiasGrads
//     }
// }

module.exports = {
    initParams,
    determineInferenceType,
    feedforward,
    getOutputLayerDelta,
    projectDeltaBackward,
    applyDerivativeAndComputeGradients,
}
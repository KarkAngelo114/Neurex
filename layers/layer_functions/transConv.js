const { red, reset, yellow } = require('../../color-code');
const activation = require('../../core/bindings');
const {cacheOutputLayerDelta, transConv, computeBiasGradsForConv, scaleDiff, transConvBackward, element_wise_mul, element_wise_sub, accumulateKernelGradsForTransConv} = require("../../core/bindings");
const { XavierInitialization, calculateTransposedTensorShape, createTensorBuffer } = require('../../utils/utils');

const initParams = (size, shape, layer_data) => {
    try {
        const useBias = layer_data.useBias;
        const [iH, iW, iD] = shape || [28, 28, 1];

        // check if the height and width values. If height (shape[0]) is 1 and width (shape[1]) is also 1, warn the user the that it should use reshape() first to properly reshape the data
        if (iH == 1 || iW == 1 ) {
            console.warn(`${yellow}[WARN]${reset} It seems you haven't reshape the data first to properly represent the data as a spatial tensor.`);
            console.warn(`${yellow}[WARN]${reset} Note that the data may be interpreted correctly, and will be propagated to subsequent layers.`);
            console.warn(`${yellow}[WARN]${reset} Incoming layer shape: [${shape}]`);
        } 
        
        const filters = layer_data.filters;
        const padding = layer_data.padding || "same";
        const [kh, kw] = layer_data.kernel_size || [3, 3];
        const strides = layer_data.strides || 1;
        const TotalSize = filters * kh * kw * iD;

        
        const fanIn = kh * kw * iD;
        const fanOut = kh * kw * filters;

        let weights = createTensorBuffer([TotalSize], {prefilledWith:'xavier', min: fanIn, max: fanOut}).data;
        let biases = useBias ? createTensorBuffer([filters], {prefilledWith:'xavier', min: fanIn, max: fanOut}).data : new Float32Array(filters);

        const weightGrads = new Float32Array(weights.length);
        const biasGrads =  new Float32Array(biases.length);


        const limit = XavierInitialization(fanIn, fanOut);

        // weights
        for (let i = 0; i < TotalSize; i++) {
            weights[i] =  (Math.random() * 2 - 1) * limit;
        }

        // biases
        if (useBias) {
            for (let i = 0; i < filters; i++) {
                biases[i] =  (Math.random() * 2 - 1) * limit;
            }
        }
        

        // calculate output shape
        const {OutputHeight, OutputWidth, CalculatedTensorShape} = calculateTransposedTensorShape(iH, iW, kh, kw, filters, strides, padding);

        // output shape and weight shape
        const outputShape = [OutputHeight, OutputWidth, filters];
        const weightShape = [filters, kh, kw, iD];
                        
        return {
            updatedSize: CalculatedTensorShape,
            updatedShape: outputShape,
            weights: weights,
            biases: biases,
            weightGrads: weightGrads,
            biasGrads: biasGrads,
            inputShape: shape,
            outputShape: outputShape,
            paramShape: weightShape,
        }
    }
    catch (error) {
        console.log(error);
        process.exit(1)
    }
    
}

const determineInferenceType = (layerObject, lossFunc, trainY) => {

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
    throw new Error(`${red}[ERROR]------- Unknown loss function: ${lossFunc}${reset}`);
}

const feedforward = (data) => {
    const layerData = data.layerData;
    const input = data.input;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;
    
    const inputShape = layerData.inputShape; // [iH, iW, iD]
    const outputShape = layerData.outputShape; // [oH, oW, oD]
    const weightShape = layerData.weightShape; // [f, kh, kw, d]
    const strides = layerData.strides;
    const filters = layerData.filters;
    const activation_function = activation[layerData.activation_function.name];

    const transConvOutput = transConv(input, inputShape, outputShape, strides, filters, weightShape, pointer, modelID);
    if (transConvOutput.some(v => Number.isNaN(v))) throw new Error("[Trans Conv Error] output array has NaNs after trans conv Ops");

    const output = activation_function(transConvOutput, pointer, modelID);
    if (output.some(v => Number.isNaN(v))) throw new Error("[Trans Conv Error] output array has NaNs after applying activation");

    layerData.cache = {
        layer_output: output,
    }

    return {
        outputs: output,
        z_values: transConvOutput,
        incrementor_value: 1
    }
}

const getOutputLayerDelta = (data) => {
    const layerObj = data.layerData;
    const preds = data.predictions;
    const actuals = data.actuals;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const lossFunc = data.loss;
    const zs = data.zs;
    const storedOutput = layerObj.cache.layer_output;
    const layerID = layerObj.layerID;


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
        const dAct = dActivation(lastLayerZs, storedOutput, pointer, modelID); 

        dOutputLayer = scaleDiff(preds, actuals, dAct);

        if (dOutputLayer.some(v => Number.isNaN(v))) throw new Error("Delta of the output layer has NaNs"); 

    }

    cacheOutputLayerDelta(dOutputLayer, pointer, modelID);

    return dOutputLayer;
   
}

const projectDeltaBackward = (data) => {
    const layerData = data.layerData;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const delta = data.delta;
    const inputShape = layerData.inputShape;
    const outputShape = layerData.outputShape;
    const weightShape = layerData.weightShape;
    const strides = layerData.strides;
    const filters = layerData.filters;
    const layerID = layerData.layerID;

    const result = transConvBackward(delta, inputShape, outputShape, strides, filters, weightShape, pointer, modelID);
    if (result.some(v => Number.isNaN(v))) throw new Error("[Trans Conv Delta Projection Error] output array has NaNs after transConvBackward() Ops");
    
    return result;
}

const applyOwnDerivative = (data) => {
    const layerData = data.layerData;
    const delta = data.delta;
    const z = data.z_value;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const layerID = layerData.layerID;

    const dActivation = activation.derivatives[layerData.activation_function.name];
    const storedOutput = layerData.cache.layer_output;

    const dAct = dActivation(z, storedOutput, pointer, modelID)
    const result = element_wise_mul(dAct, delta, pointer, modelID);
    if (result.some(v => Number.isNaN(v))) throw new Error("element_wise_mul result has NaNs in applyOwnDerivative (trans conv)");

    return result;
}

const gradientAccumulation = (data) => {
    const layerData = data.layerData;
    const deltas = data.deltas;
    const pointer = data.pointer;
    const modelID = data.modelID;
    const activation_outputs = data.activation_outputs;
    const weightGrads = data.weightGrads;
    const biasGrads = data.biasGrads;
    const layerID = layerData.layerID;

    const strides = layerData.strides;
    const filters = layerData.filters;
    const inputShape = layerData.inputShape; // [iH, iW, iD]
    const outputShape = layerData.outputShape; // [oH, oW, oD]
    const weightShape = layerData.weightShape; // [f, kh, kw, d]

    const kernelWeightGrads = accumulateKernelGradsForTransConv(activation_outputs, deltas, weightGrads, strides, filters, inputShape, outputShape, weightShape, pointer, modelID);
    if (kernelWeightGrads.some(v => Number.isNaN(v))) throw new Error("weight gradient accumulation outputs NaNs (trans conv)");
    
    const [outH, outW] = layerData.outputShape;
    const kernelBiasGrads =  computeBiasGradsForConv(biasGrads, deltas, outH, outW, filters, pointer, modelID);
    
    if (kernelBiasGrads.some(v => Number.isNaN(v))) throw new Error("bias gradient accumulation result has NaNs in accumulateBiasGradients (trans conv)");

    return {
        accumulatedWeightGrads: kernelWeightGrads,
        accumulatedBiasGrads: kernelBiasGrads
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
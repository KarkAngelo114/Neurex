const activation = require('../../core/bindings')
const { applyPadding, Convolve, ConvolveDelta, element_wise_mul, Dilate_Input, DeltaMatMul } = require("../../core/bindings");
const { XavierInitialization, calculateTensorShape, getPaddingSizes } = require("../../utils/utils");



/**
 * Initialized parameters for this layer
 * @param {Number} size number of neurons for this layer 
 * @param {Array<Number>} shape shape of the incoming input
 * @param {Object} layer_data layer_data
 * @returns {{updatedSize: Number, updatedShape: Array<Number>, weights: Float32Array, biases: Float32Array, weightGrads: Float32Array, biasGrads: Float32Array, outputTensors: Float32Array, inputShape: Array<Number>, outputShape: Array<Number>, paramShape: Array<Number>, isParametric: Boolean}}
 */
const initParams = (size, shape, layer_data) => {
    const filters = layer_data.filters;
    const [kH, kW] = layer_data.kernel_size;
    const stride = layer_data.strides || 1;
    const padding = layer_data.padding || "same";

    const inputH = shape[0];
    const inputW = shape[1];
    const inputDepth = shape[2];

    const inputShape = [inputH, inputW, inputDepth];

    const TotalSize = filters * kH * kW * inputDepth;

    let kernels = new Float32Array(TotalSize);
    let kernelGrads = new Float32Array(TotalSize);
    let biases = new Float32Array(filters);
    let biasGrads = new Float32Array(filters);
    const fanIn = kH * kW * inputDepth;
    const fanOut = kH * kW * filters;
    const limit = XavierInitialization(fanIn, fanOut);

    for (let i = 0; i < TotalSize; i++) {
        kernels[i] = (Math.random() * 2 - 1) * limit;
    }

    for (let i = 0; i < filters; i++) {
        biases[i] = (Math.random() * 2 - 1) * limit;
    }

    // Calculate output shape
    const { OutputHeight, OutputWidth, CalculatedTensorShape } = calculateTensorShape(inputH, inputW, kH, kW, filters, stride, padding);
    const output_template = new Float32Array(CalculatedTensorShape)
    // store output shape too
    const outputShape = [OutputHeight, OutputWidth, filters];

    const weightShape = [filters, kH, kW, inputDepth];
                    
    return {
        updatedSize: CalculatedTensorShape,
        updatedShape: outputShape,
        weights: kernels,
        biases: biases,
        weightGrads: kernelGrads,
        biasGrads: biasGrads,
        outputTensors: output_template,
        inputShape: inputShape,
        outputShape: outputShape,
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
    throw new Error('Convolutional layer cannot be an output layer for now');
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
    let [f, kh, kw, kd] = current_layer.weightShape;
    let [input_H, input_W, input_D] = current_layer.inputShape; 
    let padding = current_layer.padding;
    let strides = current_layer.strides;

    // 1. compute expected output tensor shape
    const { OutputHeight, OutputWidth } = calculateTensorShape(input_H, input_W, kh, kw, input_D, current_layer.strides, current_layer.padding);

    // 2. get padding sizes for each sides
    const {top, bottom, left, right} = getPaddingSizes(input_H, input_W, kh, kw, strides, padding);

    // 3. apply padding
    const {data, shape} = applyPadding(input, input_H, input_W, input_D, top, bottom, left, right);

    // 4. Perform the convolve operation using the shapes calculated in step 1
    const convolve_result = Convolve(data, current_layer.strides, [OutputHeight, OutputWidth], [f, kh, kw, kd], [shape[0], shape[1]], pointer, outputTemplatePointer);

    if (convolve_result.some(Number.isNaN)) throw new Error('NaN detected on convolve result');

    // 5. activate each depth input using the given activation function
    const activation_function = activation[current_layer.activation_function.name];
    const outputs = activation_function(convolve_result);

    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");

    return {
        outputs: outputs,
        z_values: convolve_result,
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
    throw new Error('Convolutional layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
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
    let Current_Z = zs[layer_index];
    let dActivation = activation.derivatives[current_layer.activation_function.name];
    let dL_dActivation;

    if (nextLayer.layer_name === "connected_layer") {
        const [inputSize, outputSize] = nextLayer.weightShape;
        dL_dActivation = DeltaMatMul(delta, inputSize, outputSize, pointer);
    } 
    else if (nextLayer.layer_name === "maxPooling") {
        dL_dActivation = delta;
    } 
    else if (nextLayer.layer_name === "convolutionalLayer") {
        const [Fn, KHn, KWn, KCn] = nextLayer.weightShape;
        const [oHn, oWn, oDn] = nextLayer.outputShape;
        const [oHcurr, oWcurr] = current_layer.outputShape;  // backward target shape
        const stridesN = nextLayer.strides;
        const paddingN = nextLayer.padding;

        // dilate input
        const {data: dilated, dilatedHeight: dilatedH, dilatedWidth:dilatedW} = Dilate_Input(delta, [oHn, oWn, oDn], stridesN);

        let pT, pB, pL, pR;
        if (paddingN === "valid") {
            // full conv: K-1 on every side
            pT = pB = KHn - 1;
            pL = pR = KWn - 1;
        } 
        else {
            // "same": split K-1, then top up so the result is at least oHcurr/oWcurr
            pT = Math.floor((KHn - 1) / 2); pB = (KHn - 1) - pT;
            pL = Math.floor((KWn - 1) / 2); pR = (KWn - 1) - pL;

            const needH = oHcurr + KHn - 1;          // ConvolveDelta needs Hp >= needH
            const needW = oWcurr + KWn - 1;
            const haveH = dilatedH + pT + pB;
            const haveW = dilatedW + pL + pR;
            if (haveH < needH) pB += (needH - haveH);
            if (haveW < needW) pR += (needW - haveW);
        }

        // pass the REAL dilated dims, not oHn/oWn
        const { data: PaddedInput, shape } = applyPadding(dilated, dilatedH, dilatedW, oDn, pT, pB, pL, pR);

        dL_dActivation = ConvolveDelta(PaddedInput, shape, [Fn, KHn, KWn, KCn], [oHcurr, oWcurr], pointer);
    }
                    
    const output = element_wise_mul(dActivation(Current_Z), dL_dActivation);
    if (output.some(v => Number.isNaN(v))) throw new Error("Element-wise multiplication result has NaNs");

    return {
        current_delta: output,
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
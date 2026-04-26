/**
 * This is the Layers class, each layers (except inputShape()) has their own:
 * - determineInferenceType()
 * - feedforward()
 * - getOutputLayerDelta()
 * - backpropagate()
 * - computeWeightGrads()
 * - computeBiasGrads()
 * - scaleGrads()
 *
 * Those functions are the core functions of this library to work.
 */


const {
    MatMul, 
    DeltaMatMul, 
    computeWeightGradientsForWeightsInConnectedLayer, 
    computeBiasGradsForConnected_Layer,
    applyPadding,
    Convolve, 
    Dilate_Input,
    rotate_kernels,
    ConvolveDelta,
    scaleGrads,
    element_wise_mul,
    computeBiasGradsForConv,
    MaxPool,
    ComputeGradientForKernels
} = require('../core/bindings');

const {calculateTensorShape, getPaddingSizes, ifOneHotEndcoded} = require('../utils/utils');
const activation = require('../core/bindings');
const { red, reset } = require('../color-code');


class Layers {
    constructor () {
        this.weights = [];
        this.biases = [];
        this.weightGrads = [];
        this.biaeGrads = [];
    }

    /**
     * @method inputShape
     * @param {object} shapeConfig - specify the number of features
     * @returns {Object}
     * @example
     * model.sequentialBuild([
        layer.inputShape({features: 4}),
        layer.connectedLayer("relu", 5),
        layer.connectedLayer("softmax", 3);
     ]);

     the inputShape() method allows you to get the shape of your input.
     */
    inputShape(shapeConfig) {
        try {
            if (shapeConfig.features) {
                const features = shapeConfig.features;
                this.input_shape = null;
                return {
                    layer_name: "input_layer",
                    layer_size: features,
                    input_shape: null
                };
            } else if (shapeConfig.height && shapeConfig.width && shapeConfig.depth) {
                const { height, width, depth } = shapeConfig;

                return {
                    layer_name: "input_layer",
                    layer_size: height * width * depth,
                    input_shape: [height, width, depth]
                };
            } else {
                throw new Error(`[ERROR]------- Invalid input shape config`);
            }
        } catch (error) {
            console.error(error.message);
        }
    }

    /**
     * @method connectedLayer
     * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear)
     * @param {Number} layer_size specify the number of neuron for this layer.
     * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
     * @returns {Object}
     *
     * Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will
     * build connected layers or multilayer perceptron
     */
    connectedLayer(activation_function = 'relu', layer_size = 5) {
        try {

            if (!activation_function || !layer_size || layer_size <= 0) {
                throw new Error(`[ERROR]------- Layer Error | Activation function: ${activation_function} | layer size: ${layer_size}`);
            }

            let function_name = activation_function.toLowerCase();

            if (!activation[function_name] || !activation.derivatives[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                "layer_name":"connected_layer", 
                "activation_function":activation[function_name], 
                "derivative_activation_function":activation.derivatives[function_name],
                "layer_size":layer_size,
                determineInferenceType: (layerObject, lossFunc, trainY) => {
                    
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

                    if (activation_function === "linear" && (lossFunc === "mae" || lossFunc === "mse")) {
                        return "regression";
                    }

                    if ((activation_function === "sigmoid" || activation_function === "tanh") && (lossFunc === "binary_cross_entropy")) {
                        return "binary_classification";
                    }

                    if (activation_function === "softmax" && (lossFunc === "categorical_cross_entropy" || lossFunc === "sparse_categorical_cross_entropy")) {
                        return "multi_class_classification";
                    }

                    //  if none satisfies the conditions above, throw an error
                    throw new Error(`${red}[ERROR]------- Using ${lossFunc} having output size of ${layer_size} and an ${activation_function} function in the output layer is currently unavailable for this core's task.${reset}`);
                },
                feedforward: (onGPU, input, weights, biases, current_layer) => {
                    
                    const z_values = MatMul(input, weights, biases, current_layer.weightShape[0], current_layer.weightShape[1]);
                    const activation_function = activation[function_name];

                    let outputs = activation_function(z_values);
                    
                    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");
                    
                    return {
                        outputs, 
                        z_values,
                        incrementor_value: 1
                    };
                },
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
                    let dActivation = activation.derivatives[function_name];
                    let dOutputLayer = new Float32Array(preds.length); 

                    if (tasktype === "binary_classification" || (tasktype === "multi_class_classification" && lossFunc === "categorical_cross_entropy")) {
                        for (let i = 0; i < dOutputLayer.length; i++) {
                            dOutputLayer[i] = preds[i] - actuals[i];
                        }
                    }
                    else if (tasktype === "multi_class_classification" && lossFunc === "sparse_categorical_cross_entropy") {
                        dOutputLayer.set(preds);
                        dOutputLayer[actuals[0]] -= 1;
                        
                    }
                    else if (tasktype === "regression") {
                        const lastLayerZs = zs[zs.length - 1]; 
                        const dAct = dActivation(lastLayerZs); 
                        
                        for (let i = 0; i < dOutputLayer.length; i++) {
                            dOutputLayer[i] = (preds[i] - actuals[i]) * dAct[i];
                        }
                    }

                    return dOutputLayer;
                },
                backpropagate: (onGPU, next_weights, next_delta, zs, layer_index, current_layer, allWeights, activations, nextLayer) => {
                    const dActivation = activation.derivatives[function_name];
                    const dAct = dActivation(zs[layer_index]);
                    const delta_res = DeltaMatMul(next_delta, next_weights, nextLayer.weightShape[0], nextLayer.weightShape[1]);
                    const current_delta = element_wise_mul(dAct, delta_res);     

                    if (current_delta.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");            

                    return {
                        current_delta,
                        decrementor_value: 1
                    };
                },
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => computeWeightGradientsForWeightsInConnectedLayer(activation_outputs, deltas, weightGrads, layer_data.weightShape[0], layer_data.weightShape[1]),
                computeBiasGradients: (biasgrads, deltas, layer_data) => computeBiasGradsForConnected_Layer(biasgrads, deltas),
                scaleGrads: (grads, batchSize, layer_data) => scaleGrads(grads, batchSize)
            };
        }
        catch (error) {
            console.log(error.message);
        }
    }

    /**
     * 
     * @method convolutionalLayer
     * @param {Number} filters - the number of filters for this convolutional layer. Produces the same number of output features
     * @param {Number} strides - It determines how much the filter overlaps with the input as it slides across.
     * @param {Array<Number>} kernel_size - the size of the kernel (or filter) that will slide and extracts input features
     * @param {String} activation_function - the activation function to be use for this layer
     * @param {String} padding - adds extra values (typically 0s) around the border of an input before applying a convolutional filter
     * @throws {Error} - if any of the parameters are invalid.
     * @returns {Object}
     *
     * Allows you to add convolutional layers in your model architecture in sequential building.
     */
    convolutionalLayer(filters = 3, strides = 1, kernel_size = [3, 3], activation_function = 'relu', padding = 'same') {
        try {
            if (!filters || filters <= 0) throw new Error(`[ERROR]-------- Filters cannot be empty, less than or equal to 0. Filters: ${filters}`);
            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);
            if (!kernel_size || kernel_size.length == 0 || (kernel_size[0] <= 0 || kernel_size[1] <= 0)) throw new Error(`[ERROR]------- Kernels cannot be empty, nor it's height or width is less than or equal to 0. Kernel size: ${kernel_size}`);
            if (!activation_function || activation_function == undefined || activation_function == null || activation_function === "") throw new Error(`[ERROR]-------- activation_function cannot be empty, null or undefined.`);
            if (!padding || padding == undefined || padding == null || padding === "") throw new Error(`[ERROR]-------- Padding cannot be empty, null or undefined.`);

            // check if the padding is same/valid, otherwise throw error
            let paddings = ["same", "valid"];
            if (!paddings.includes(padding.toLowerCase())) {
                throw new Error(`[ERROR]------- ${padding.toLowerCase()} is invalid. Use 'same' or 'valid' only`);
            }

            // check if the activation function is valid
            const function_name = activation_function.toLowerCase();

            if (!activation[function_name] || !activation.derivatives[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                "layer_name":"convolutionalLayer",
                "activation_function":activation[function_name],
                "derivative_activation_function":activation.derivatives[function_name],
                "kernel_size":kernel_size,
                "filters":filters,
                "padding":padding.toLowerCase(),
                "strides":strides,
                determineInferenceType: (layerObject, lossFunc, trainY) => {

                    if (lossFunc === "categorical_cross_entropy" && activation_function === "softmax") {
                        // check if the trainY are one hot encoded. Categorical Cross Entropy works wiht one-hot encoded labels
                        const isOneHotEncoded = ifOneHotEndcoded(trainY);
                        if (!isOneHotEncoded) throw new Error("Labels must be one hot encoded if the loss function is 'categorical_cross_entropy' and the activation function is `softmax`.");
                    }

                    if (activation_function === "linear" && (lossFunc === "mae" || lossFunc === "mse")) {
                        return "regression";
                    }

                    if ((activation_function === "sigmoid" || activation_function === "tanh") && (lossFunc === "binary_cross_entropy")) {
                        return "binary_classification";
                    }

                    if (activation_function === "softmax" && (lossFunc === "categorical_cross_entropy" || lossFunc === "sparse_categorical_cross_entropy")) {
                        return "multi_class_classification";
                    }

                    /**
                    * Convolution layers might have it's on way of determining task, I'll leave this as one of my TO DOs
                    */
                },
                feedforward: (onGPU, input, weights, biases, current_layer) => {
                    
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
                    const convolve_result = Convolve(data, weights, biases, current_layer.strides, OutputHeight, OutputWidth, f, kh, kw, kd, shape[0], shape[1]);

                    if (convolve_result.some(Number.isNaN)) throw new Error('NaN detected on convolve result');

                    // 5. activate each depth input using the given activation function
                    const activation_function = activation[function_name];

                    const outputs = activation_function(convolve_result);

                    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has Nans");

                    return {
                        outputs: outputs,
                        z_values: convolve_result,
                        incrementor_value: 1
                    };
                },
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
                    /**
                    * Convolution layers has different process of getting delta of the output layer, so this is another TO DOs, but for now, throw an error 
                    */

                    throw new Error('Convolutional layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
                    process.exit(1);
                    // let dOutputLayer = new Float32Array(preds.length); 

                    // return dOutputLayer;
                },
                backpropagate: (onGPU, next_weights, next_delta, zs, layer_index, currentLayer, weights, activations, next_layer, allLayers) => {
                    let Current_Z = zs[layer_index];
                    let dActivation = activation.derivatives[function_name];
                    let dL_dActivation;

                    if (next_layer.layer_name === "connected_layer") {
                        const [inputSize, outputSize] = next_layer.weightShape;
                        dL_dActivation = DeltaMatMul(next_delta, next_weights, inputSize, outputSize);
                    } 
                    else if (next_layer.layer_name === "maxPooling") {
                        dL_dActivation = next_delta;
                    } 
                    else if (next_layer.layer_name === "convolutionalLayer") {
                        const [Fn, KHn, KWn, KCn] = next_layer.weightShape;
                        const [oHn, oWn, oDn] = next_layer.outputShape;
                        const [oHcurr, oWcurr] = currentLayer.outputShape;  // backward target shape
                        const stridesN = next_layer.strides;
                        const paddingN = next_layer.padding;

                        const rotated = rotate_kernels(next_weights, Fn, KHn, KWn, KCn);
                        const dilated = Dilate_Input(next_delta, [oHn, oWn, oDn], stridesN);
                        
                        const dilatedH = oHn * stridesN + (oHn - 1) * (stridesN - 1);
                        const dilatedW = oWn * stridesN + (oWn - 1) * (stridesN - 1);

                        let pT, pB, pL, pR;
                        if (paddingN === "valid") {
                            // full conv: K-1 on every side
                            pT = pB = KHn - 1;
                            pL = pR = KWn - 1;
                        } else {
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
                        const { data, shape } = applyPadding(dilated, dilatedH, dilatedW, oDn, pT, pB, pL, pR);

                        dL_dActivation = ConvolveDelta(data, shape, rotated, [Fn, KHn, KWn, KCn], oHcurr, oWcurr);
                    }
                    
                    const output = element_wise_mul(dActivation(Current_Z), dL_dActivation);
                    if (output.some(v => Number.isNaN(v))) throw new Error("Element-wise multiplication result has NaNs");

                    return {
                        current_delta: output,
                        decrementor_value: 1
                    };
                    
                    
                    
                },
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => {
                    const [filters, kH, kW, inDepth] = layer_data.weightShape
                    const [inH, inW] = layer_data.inputShape
                    const [outH, outW] = layer_data.outputShape


                    const output = ComputeGradientForKernels(activation_outputs,deltas,weightGrads,inH, inW, inDepth,outH, outW, filters,kH, kW);

                    if (output.some(Number.isNaN)) throw new Error(`Has NaNs after accumulation of kernel grads`);

                    return output;
                },
                computeBiasGradients: (biasgrads, deltas, layer_data) => {
                    const [filters] = layer_data.weightShape;
                    const [outH, outW] = layer_data.outputShape;

                    return computeBiasGradsForConv(biasgrads, deltas, outH, outW, filters);
                },
                scaleGrads: (grads, batchSize, layer_data) => scaleGrads(grads, batchSize)
            }
        }
        catch (error) {
            console.error(error);
            process.exit(1);
        }
    }

    /**
     * @method maxPooling
     * @param {Array<Number>} poolSize - determines the pool size window 
     * @param {Number} strides - It determines how much the pool window slides across the input tensor.
     * @param {String} padding - `same` or `valid`
     * @throws {Error} - if any of the values are 0s or negative for the pool size and strides or the padding is invalid
     *
     * `maxPooling` is use for downsampling operation that reduces the spatial dimensions of an input tensor by taking the maximum value over a defined sliding window
     */
    maxPooling(poolSize, strides = 1, padding = "same") {
        try {
            if (poolSize[0] <= 0 || poolSize[1] <= 0) {
                throw new Error(`[ERROR]------- pool size value cannot be 0 or a negative value`);
            }

            // check if the padding is same/valid, otherwise throw error
            let paddings = ["same", "valid"];
            if (!paddings.includes(padding.toLowerCase())) {
                throw new Error(`[ERROR]------- ${padding.toLowerCase()} is invalid. Use 'same' or 'valid' only`);
            }

            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);

            return {
                "layer_name":"maxPooling",
                "poolSize": poolSize,
                "padding": padding,
                "strides":strides,
                determineInferenceType: (layerObject, lossFunc, trainY) => {
                    throw new Error('Max pooling layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
                    process.exit(1);
                },
                feedforward: (onGPU, input, weights=null, biases=null, current_layer) => {
                    const [inputh, inputw, inputd] = current_layer.inputShape;
                    const [outputh, outputw, outputd] = current_layer.outputShape;
                    const [poolHeight, poolWidth] = current_layer.poolSize;
                    const strides = current_layer.strides;
                
                    let {output, maxIndices} = MaxPool(input, [poolHeight, poolWidth], [inputh, inputw, inputd], [outputh, outputw, outputd], strides);
                    current_layer.maxIndices = maxIndices;

                    if (output.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");

                    return {
                        outputs:output,
                        z_values: output,
                        incrementor_value:0
                    }
                },
                getOutputLayerDelta: (preds, actuals, zs, lossFunc, tasktype, layerObj) => {
                    /**
                    * Max pooling has different process of getting delta of the output layer, so this is another TO DOs, but for now, throw an error 
                    */

                    throw new Error('Max pooling layer cannot be an output layer for now. Consider use a connected layer as its classifier head');
                    process.exit(1);
                },
                backpropagate: (onGPU, next_weights = null, prev_delta, zs, layer_index, currentLayer, weights, activations, next_layer, allLayers) => {
                    let next_delta = prev_delta;
                    const [inputH, inputW, inputD] = currentLayer.inputShape;
                    const [outputH, outputW, outputD] = currentLayer.outputShape;
                    const [poolHeight, poolWidth] = currentLayer.poolSize;
                    const strides = currentLayer.strides;
                    const padding = currentLayer.padding;

                    if (next_layer.layer_name === "connected_layer") {
                        const [inputSize, outputSize] = next_layer.weightShape;
                        next_delta = DeltaMatMul(prev_delta, next_weights, inputSize, outputSize);
                    }

                    const delta = new Float32Array(inputH * inputW * inputD);
                    const indices = currentLayer.maxIndices;

                    for (let i = 0; i < next_delta.length; i++) {
                        let idx = indices[i];
                        delta[idx] += next_delta[i];
                    }

                    
                    return {
                        current_delta: delta,
                        decrementor_value:0
                    }
                },
                computeWeightGradients: (activation_outputs, deltas, weightGrads, layer_data) => {
                    // max pooling layer has no params like weights and biases, so no functions here :)
                },
                computeBiasGradients: (biasgrads, deltas, layer_data) => {
                    // max pooling layer has no params like weights and biases, so no functions here :)
                },
                scaleGrads: () => {
                    // max pooling layer has no params like weights and biases, so no functions here :)
                },
            }
        }
        catch (error) {
            console.error(error);
            process.exit(1);
        }
    }
}

module.exports = Layers;
/**
 * This is the Layers class, each layers (except inputShape()) has it's own:
 * - feedforward()
 * - backpropagate()
 * - computeWeightGrads()
 * - computeBiasGrads()
 * - scaleGrads()
 *
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
    ComputeGradientForKernels
} = require('../core/bindings');

const {calculateTensorShape, getPaddingSizes} = require('../utils/utils');
const activation = require('../core/bindings');


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
     * Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will
     * build connected layers or multilayer perceptron
     * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear)
     * @param {Number} layer_size specify the number of neuron for this layer.
     * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
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
                feedforward: (onGPU, current_input, weights, biases, current_layer) => {

                    let input = current_input;

                    const z_values = MatMul(input, weights, biases, current_layer.weightShape[0], current_layer.weightShape[1]);
                    const activation_function = activation[function_name];

                    let outputs;
                    outputs = activation_function(z_values);
                    
                    if (outputs.some(v => Number.isNaN(v))) throw new Error("Error - output array has NaNs");
                    
                    return {
                        outputs, 
                        z_values,
                        incrementor_value: 1
                    };
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

                // since all are in float32 1D array, we can use the function for scaling weights and biases gradients
                scaleGrads: (grads, batchSize, layer_data) => scaleGrads(grads, batchSize)
            };
        }
        catch (error) {
            console.log(error.message);
        }
    }

    /**
     * 
     * Allows you to add convolutional layers in your model architecture in sequential building.
     * @method convolutionalLayer
     * @param {Number} filters - the number of filters for this convolutional layer. Produces the same number of output features
     * @param {Number} strides - It determines how much the filter overlaps with the input as it slides across.
     * @param {Array<Number>} kernel_size - the size of the kernel (or filter) that will slide and extracts input features
     * @param {String} activation_function - the activation function to be use for this layer
     * @param {String} padding - adds extra values (typically 0s) around the border of an input before applying a convolutional filter
     * @throws {Error} - if any of the parameters are invalid.
     *
     */
    convolutionalLayer(filters = 3, strides = 1, kernel_size = [3, 3], activation_function = 'relu', padding = 'same') {
        try {
            if (!filters || filters <= 0) throw new Error(`[ERROR]-------- Filters cannot be empty, less than or equal to 0. Filters: ${filters}`);
            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);
            if (!kernel_size || kernel_size.length == 0 || (kernel_size[0] <= 0 || kernel_size[1] <= 0)) throw new Error(`[ERROR]------- Kernels cannot be empty, nor it's height or width is less than or equal to 0. Kernel size: ${kernel_size}`);
            if (!activation_function || activation_function == undefined || activation_function == null || activation_function === "") throw new Error(`[ERROR]-------- activation_function cannot be empty, null or undefined.`);
            if (!padding || padding == undefined || padding == null || padding === "") throw new Error(`[ERROR]-------- Padding cannot be empty, null or undefined.`);

            // check if the padding is valid
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
                // kernels are also considered weights
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

                backpropagate: (onGPU, next_weights, next_delta, zs, layer_index, currentLayer, weights, activations, next_layer, allLayers) => {
                    let input = next_delta;
                    let params = next_weights;
                    let Current_Z = zs[layer_index];
                    let dActivation = activation.derivatives[function_name];

                    if (next_layer.layer_name === "connected_layer") {
                        const [inputSize, outputSize] = next_layer.weightShape;
                        input = DeltaMatMul(input, params, inputSize, outputSize);

                        if (input.some(v => Number.isNaN(v))) throw new Error('Delta coming after DeltaMatMuk() has NaNs'); 
                        
                    }

                    const kernels = weights[layer_index];
                    const [f, kh, kw, kc] = currentLayer.weightShape; 
                    const [iH, iW, iD] = currentLayer.inputShape;
                    const [oH, oW, oD] = currentLayer.outputShape;
                    const strides = currentLayer.strides;
                    const padding = currentLayer.padding;

                    // rotate kernels
                    const rotated_kernels = rotate_kernels(kernels, f, kh, kw, kc);

                    // dilate delta
                    const dilated = Dilate_Input(input, [oH, oW, oD], strides);
                    if (dilated.some(v => Number.isNaN(v))) throw new Error("Input dilation has NaNs");

                    let padTop, padBottom, padLeft, padRight;

                    if (padding === "valid") {
                        // FULL padding for backprop
                        padTop = kh - 1 + 1;
                        padBottom = kh - 1 + 1;
                        padLeft = kw - 1 + 1;
                        padRight = kw - 1 + 1;
                    } else if (padding === "same") {
                        const padAlongHeight = kh - 1;
                        const padAlongWidth  = kw - 1;

                        padTop = Math.floor(padAlongHeight / 2);
                        padBottom = padAlongHeight - padTop;

                        padLeft = Math.floor(padAlongWidth / 2);
                        padRight = padAlongWidth - padLeft;
                    }

                    // apply padding
                    const {data, shape} = applyPadding(dilated, oH, oW, oD, padTop, padBottom, padLeft, padRight);
                    if (data.some(v => Number.isNaN(v))) throw new Error("Padded input has NaNs");


                    // perform delta convolution
                    const delta_res = ConvolveDelta(data, shape, rotated_kernels, [f, kh, kw, kc], oH, oW);
                    if (delta_res.some(v => Number.isNaN(v))) throw new Error("Delta convolution result has NaNs");

                    // console.log(delta_res.length);
                    // console.log(Current_Z.length);

                    // perform element-wise multiplication with derivative outputs of Zs and the delta convolution results
                    const output = element_wise_mul(dActivation(Current_Z), delta_res);
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
                
                // since all are in float32 1D array, we can use the function for scaling weights and biases gradients
                scaleGrads: (grads, batchSize, layer_data) => scaleGrads(grads, batchSize)


            }

        }
        catch (error) {
            console.error(error);
        }
    }
}

module.exports = Layers;
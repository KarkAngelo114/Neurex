
const activation = require('../gpu/kernels/activations');
const {computeForward, computeBackprop} = require('../gpu/kernels/matrixMultiplication');
const {convolve} = require('../gpu/kernels/convolutionalKernel');

/**
 * 
 * @class
 *
 * Stacking layers will return the layer's information such as the layer_name, activation_function, layer_size, kernel_size (for convolutional), etc.
 * available layers:
 * inputShape() - This will tell the network that your input layer has this X number of input neuron.
 * connectedLayer() - to build fully connected layers. For building ANNs
 */
class Layers {
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
    * 
    *
    @method flatten()  
    Flattens the output of the last convolutional layer into a 1D array. This layer is crucial before connecting to fully connected layers as
    it bridges the gap between feature extraction part of your network and the connected layers.
    */
    flatten() {
        return {
            "layer_name":"flatten_layer",
            feedforward: (onGPU, input, weights=null, bias = null) => {
                
                const flattened = input.flat(Infinity);

                return {
                    outputs: flattened,
                    z_values: flattened,
                    incrementor_value: 0
                }
            },
            backpropagate: (onGPU) => {

                return {
                    decrementor_value: 0
                }
            }
        }
    }


    /**
     * Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will
     * build connected layers or multilayer perceptron
     * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear)
     * @param {Number} layer_size specify the number of neuron for this layer.
     * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
     */
    connectedLayer(activation_function, layer_size) {
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
                feedforward: (onGPU, input, weights, biases) => {
                    // All the logic for matrix multiplication and activation
                    const z_values = computeForward(onGPU, input, weights, biases);
                    const activation_function = activation[function_name];
                    let outputs;

                    if (activation_function.name === "softmax") {
                        outputs = activation_function(z_values); // Apply softmax to all z_values
                    } else {
                        // If GPU not available, then perform neuron-by-neuron for getting the activated output
                        if (!this.onGPU) {
                            outputs = [];
                            for (let i= 0; i < biases.length; i++) {
                                outputs.push(activation_function(z_values[i]));
                            }
                        }
                        else {
                            // if GPU available, shove the dot products (z-values or pre-activated outputs) to compute the activated outputs for every neurons
                            outputs = activation_function(z_values, this.onGPU);
                        }
                    }
                    return {
                        outputs, 
                        z_values,
                        incrementor_value: 1
                    };
                },
                backpropagate: (onGPU, next_weights, next_delta, zs, layer_index) => {
                    const weighted_delta = computeBackprop(onGPU, next_weights, next_delta);
                    const current_delta = weighted_delta.map((value, i) =>
                        value * activation.derivatives[function_name](zs[layer_index][i])
                    );
                    return {
                        current_delta,
                        decrementor_value: 1
                    };
                }
            };
        }
        catch (error) {
            console.log(error.message);
        }
    }

    /**
     * 
     * Allows you to add convolutional layers in your model architecture in sequential building.
     * @method convolutional2D
     * @param {Number} filters - the number of filters for this convolutional layer. Produces the same number of output features
     * @param {Number} strides - It determines how much the filter overlaps with the input as it slides across.
     * @param {Array<Number>} kernel_size - the size of the kernel (or filter) that will slide and extracts input features
     * @param {String} activation_function - the activation function to be use for this layer
     * @param {String} padding - adds extra values (typically 0s) around the border of an input before applying a convolutional filter
     * @throws {Error} - if any of the parameters are invalid.
     *
     */
    convolutional2D(filters, strides, kernel_size, activation_function, padding) {
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
                "layer_name":"convolutional2D",
                "activation_function":activation[function_name],
                "derivative_activation_function":activation.derivatives[function_name],
                "kernel_size":kernel_size,
                "filters":filters,
                "padding":padding,
                // kernels are also considered weights
                feedforward: (onGPU, input, weights, biases) => {
                    // Standard behavior: Combine all input feature maps into a single 3D tensor
                    // This assumes the input is a batch of samples.
                    // If input.length is 1, it's a single sample.
                    let single_sample_input;
                    if (input.length > 0 && input[0].length > 0 && input[0][0].length > 1) {
                        // Input is already a multi-channel tensor, which is the expected case
                        single_sample_input = input[0];
                    } else {
                        // Combine the array of feature maps into a single tensor with depth
                        const height = input[0].length;
                        const width = input[0][0].length;
                        const depth = input.length;
                        single_sample_input = new Array(height).fill(null).map(() =>
                            new Array(width).fill(null).map(() => new Array(depth))
                        );

                        for (let d = 0; d < depth; d++) {
                            for (let h = 0; h < height; h++) {
                                for (let w = 0; w < width; w++) {
                                    single_sample_input[h][w][d] = input[d][h][w][0];
                                }
                            }
                        }
                    }
                    
                    const new_feature_maps = convolve(onGPU, filters, strides, single_sample_input, weights, biases, padding);

                    const outputs = [];
                    const z_values = new_feature_maps; // z_values are the new feature maps before activation
                    let activation_function = activation[function_name];

                    new_feature_maps.forEach(featureMap => {
                        const activatedMap = featureMap.map(row =>
                            row.map(cell => [activation_function(cell[0], false)])
                        );
                        outputs.push(activatedMap);
                    });

                    return {
                        outputs,
                        z_values,
                        incrementor_value: 1
                    };
                },

                backpropagate: (onGPU) => {
                    return {
                        decrementor_value: 1
                    }
                }
            }

        }
        catch (error) {
            console.error(error);
        }
    }
}

module.exports = Layers;
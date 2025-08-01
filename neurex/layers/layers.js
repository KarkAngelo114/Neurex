
const activations = require('../gpu/kernels/activations');
const activation = require('../gpu/kernels/activations');
const {computeForward, computeBackprop} = require('../gpu/kernels/matrixMultiplication');

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

    flattens a 2D matrix into 1D array
    Example:

    first row:
    [[x1, x2, x3], [x1, x2, x3], [x1, x2, x3], [x1, x2, x3], [x1, x2, x3], [x1, x2, x3]]

    flattened:

    [x1, x2, x3, x4, x5, x6, ....]
    */
    flatten() {
        return {"layer_name":"flatten_layer"}
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
                throw new Error(`Activation function '${funcName}' or its derivative not found`);
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
                    return { outputs, z_values };
                },
                backpropagate: (onGPU, next_weights, next_delta, zs, layer_index) => {
                    const weighted_delta = computeBackprop(onGPU, next_weights, next_delta);
                    const current_delta = weighted_delta.map((value, i) =>
                        value * activation.derivatives[function_name](zs[layer_index][i])
                    );
                    return current_delta;
                }
            };
        }
        catch (error) {
            console.log(error.message);
        }
    }
}

module.exports = Layers;
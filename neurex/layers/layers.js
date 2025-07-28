
const activation = require('../gpu/kernels/activations');

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
     * @param {Array} data - the dataset

     the inputShape() method allows you to get the shape of your input.
     This will tell the network that your input layer has this X number of input neuron.
     Ensure that your dataset has no missing values, otherwise perform data cleaning.
     */
    inputShape(data){
        let size = data[0].length;
        // this refers to the first array of features and the number of features is the number of input neurons in the input layer
        // and assumes that the subsequent data has the same number of features
        // this is why it is important to perform data preprocessing at first before feeding to avoid having problems.

        // do a loop to all the rows to check for shape inconsistencies
        try {
            data.forEach((rows, i) => {
                if (rows.length != size) {
                    throw new Error(`[ERROR]------- Shape mismatch on row${i}. Ensure that all shapes has the same shape.\nExpected shape: ${size}. Row${i} has ${rows.length}.`);
                }
            });

            return {"layer_name":"input_layer", "layer_size":size};
        }
        catch (error) {
            console.log(error);
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
                throw new Error(`Activation function '${funcName}' or its derivative not found`);
            }

            return {"layer_name":"connected_layer", "activation_function":activation[function_name], "derivative_activation_function":activation.derivatives[function_name],"layer_size":layer_size};
        }
        catch (error) {
            console.log(error.message);
        }
    }
}

module.exports = Layers;
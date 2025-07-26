
/**
 * 

 This interpreter is dedicated for Neurex

 */


const activation = require('../activations');
const zlib = require('zlib');
const fs = require('fs');
const path = require('path');



/**
 * This class allows you to run inference predictions on your applications. You can load your trained model and run predictions
 *
 * @class
 */
class Interpreter {
    constructor () {
        this.weights = []; // weights for calculating the dot product of each nueron
        this.biases = []; // biases for calculating the dot product of each nueron
        this.num_layers = 0; // the number layers in a network (hidden and the output layer) 
        this.activation_functions = []; // activation functions for each layer
        this.number_of_neurons = []; // the number of neurons on each layer. Stored in array
        this.input_size = 0; // the size of the input layer, basically the number of input neurons.
    }

    /**
    * @method loadSavedModel()
    * @param {*} model - the trained model

    The loadSavedModel() method allows you to load the trained model. The model is typically in .nrx file format which contains the learned parameters of your trained model

    */
    loadSavedModel(model) {
        try {
            if (!model) {
                throw new Error("No model provided");
            }

            const dir = path.dirname(require.main.filename);
            const model_file = path.join(dir, `${model}`);

            // Check extension
            if (path.extname(model_file) !== '.nrx') {
                throw new Error("Invalid file type. Only .nrx model files are supported.");
            }

            // Read file
            const rawBuffer = fs.readFileSync(model_file);

            // Validate magic header
            const header = rawBuffer.slice(0, 4).toString('utf-8');
            if (header !== 'NRX1') {
                throw new Error("Invalid file format. Not a valid NRX model.");
            }

            // Check version
            const version = rawBuffer[4];
            if (version !== 0x01) {
                throw new Error(`Unsupported NRX version: ${version}`);
            }

            // Decompress and parse
            const compressedData = rawBuffer.slice(5);
            const jsonString = zlib.inflateSync(compressedData).toString('utf-8');
            const modelData = JSON.parse(jsonString);

            // Assign properties
            this.task = modelData.task;
            this.loss_function = modelData.loss_function;
            this.epoch_count = modelData.epoch;
            this.batch_size = modelData.batch_size;
            this.optimizer = modelData.optimizer;
            this.learning_rate = modelData.learning_rate;
            this.activation_functions = modelData.activation_functions.map(name => activation[name.toLowerCase()]);
            this.derivative_functions = modelData.derivative_functions.map(name => activation.derivatives[name.toLowerCase()]);
            this.input_size = modelData.input_size;
            this.output_size = modelData.output_size;
            this.num_layers = modelData.num_layers;
            this.number_of_neurons = modelData.number_of_neurons;
            this.weights = modelData.weights;
            this.biases = modelData.biases;

            console.log(`[SUCCESS]------- Model ${model} successfully loaded`);
        } catch (error) {
            console.log(error.message);
        }
    }

    /**
     * 

    @method modelSummary()

    Shows the model architecture
     */
    modelSummary() {
        console.log(`Input size: ${this.input_size}`);
        console.log(`Number of layers: ${this.num_layers}`);
        console.log('Layer\tNeurons\tActivation');
        for (let i = 0; i < this.num_layers; i++) {
            const actFn = this.activation_functions[i];
            // Try to get the function name, fallback to 'custom' if not available
            const actName = actFn && actFn.name ? actFn.name : (typeof actFn === 'string' ? actFn : 'custom');
            console.log(`${i + 1}\t${this.number_of_neurons[i]}\t${actName}`);
        }
    }

    /**
     * 
     @method predict()
     @param {Array} input - input data 
     @returns Array of predictions
     @throws Error when there's shape mismatch and no input data

     produces predictions based on the input data
     */
    predict(input) {
        try {
            if (!input) {
                throw new Error("\n[ERROR]-------No inputs")
            }

            if (input[0].length != this.input_size) {
                throw new Error(`\n[ERROR]-------Shape Mismatch | Input shape length: ${input[0].length} | Expecting ${this.input_size}`);
            }

            let outputs = [];
            for (let sample_index = 0; sample_index < input.length; sample_index++) {
                /**
                 * 
                 we loop through the entire loaded dataset
                 */
                const array_of_features = input[sample_index];
                // perform feedforward, similar when training but only outputs predictions. No updating of weights and biases.
                const {predictions} = this.#Feedforward(array_of_features);
                outputs.push(predictions);
            }
            return outputs;
        }
        catch (err) {
            console.error(err.message);
        }
    }

    #Feedforward(input) {
        let current_input = input
        let all_layer_outputs = [input];
        let zs = [];

        
        // first outer loop: getting the layers of the network
        /**
            when calling construct_layer(), the this.num_layers adds up.

            assume we have contructed only 2 layers (1 hidden layer and 1 output layer)
            therefore this for loop will interate 2 times to perform operations inside
            and the value of "layer" will specify what index in the this.weights and this.biases array going to use
            and also the number of neurons stored in the this.number_of_neurons array
         */
        for (let layer = 0; layer < this.num_layers; layer++) {
            /** get the array of biases from the array of arrays of biases 

                biases : [
                    [0b1, 0b2, 0b3, ...], <- index 0
                    [1b1, 1b2, 1b3, ...], <- index 1
                    [2b1, 2b2, 2b3, ...], <- index 2
                    [3b1, 3b2, 3b3, ...], <- index 3
                    ....
                ]

                assumes:
                    let layer = 0;
                    
                    then:
                    biases : [
                        [0b1, 0b2, 0b3, ...], <- index 0 - these are the biases we get for this layer
                        [1b1, 1b2, 1b3, ...], 
                        [2b1, 2b2, 2b3, ...], 
                        [3b1, 3b2, 3b3, ...], 
                        ....
                    ]
            */
            const layer_biases = this.biases[layer];

            /** get the array of weights from the array of arrays of weights 

                weights : [
                    [[0weights1],[0weights2],[0weights3], ...], <- index 0
                    [[2weights1],[2weights2],[2weights3], ...], <- index 1
                    [[3weights1],[3weights2],[3weights3], ...], <- index 2
                    ....
                ]

                assumes:
                    let layer = 0;
                    
                    then:
                    weights : [
                        [[0weights1],[0weights2],[0weights3], ...], <- index 0 - these are the weights for this layer and each sub arrays are the weights for neuron connections
                        [[2weights1],[2weights2],[2weights3], ...], 
                        [[3weights1],[3weights2],[3weights3], ...],
                        ....
                    ]
            */
            const layer_weights = this.weights[layer];

            /**
             * in the constructor, we have "this.number_of_neurons" which is an array. This is because when constructing the network using 
            construct_layer(activation_func, layer_size), the layer_size is appended to the "this.number_of_neurons". And since we are in 
            the for-loop for layer, we can access how many neurons that this layer composes and we can get it in the array.

            number_of_neurons = [
                    7, <- index 0
                    7, <- index 1
                    5, <- index 2
                    ....
            ]

            assumes: 
                let layer = 0;

                then: 
                    number_of_neurons = [
                        7, <- index 0 - this layer has 7 neurons and we can use it to get the arrays of weights in the array of arrays of weights for this layer
                        7, 
                        5, 
                        ....
                ]
             */
            const num_neurons = this.number_of_neurons[layer]; 

            /**
             * in the cunstructor, we have "this.activation_functions". This is because when constructing the network using 
            construct_layer(activation_func, layer_size), the activation_func is appended to the this.activation_functions.
            And since we are in the for-loop for layers, we can use the layer(int) value to index what activation function is
            assigned for this layer.

            activation_functions = [
                "relu", <- index 0
                "relu", <- index 1
                "linear", <- index 2
                ....
            ]

            assumes:
                let layer = 0
                        
                then: 

                    activation_functions = [
                        "relu", <- index 0 - this is the activation function going to be use by all neurons in this layer
                        "relu", 
                        "linear", 
                        ....
                    ]
            */
            const activation_function = this.activation_functions[layer];

            let outputs = [];
            let z_values = [];

            // main loop: loop over neurons in the layer
            /**
                calculates the dot product for each current input
                the outer loop is the neuron itself. If there are 7 neurons in this current layer, 
                it will loop 7 times and the innermost for loop is the calculation of dot product

             */
            for (let neuron = 0; neuron < num_neurons; neuron++) {
                let dot_product_output = 0; // the dot product after the innermost for loop is done calculating the datapoints inside the current_input array

                /**
                    assumes:
                        let i = 0
                            current_input = [
                                [x1, x2, x3, ...], <- index 0
                                ....
                            ]
                        then:
                            current_input = [x1, x2, x3, ...] <- each feature will be calculated inside this neuron using a dot-product
                            solution:
                                x = (x1 * 0weights1) + (x2* 0weights2) + (x3 * 0weights3) .... 
                    
                 */
                for (let i = 0; i < current_input.length; i++) {
                    dot_product_output += current_input[i] * layer_weights[i][neuron];
                }
                /**
                
                    Once we get the dot-product, we apply the bias for this neuron
                    Assume:
                        let layer = 0;
                        let neuron = 0;

                    then:
                        biases : [
                            [
                                0b1, <- index 0 - this is the bias we get for this neuron
    These are the biases        0b2, 
        for this layer          0b3, 
                                ...
                            ], 
                            [
                                1b1, 
                                1b2, 
                                1b3, 
                                ...
                            ], 
                            ....
                        ]
                 */
                z_values.push(dot_product_output + layer_biases[neuron]); // Store z-value
            }

            // After computing all z_values for the current layer
            if (activation_function.name === "softmax") {
                outputs = activation_function(z_values); // Apply softmax to all z_values
            } else {
                // For other activations, apply individually
                for (let j = 0; j < num_neurons; j++) {
                    outputs.push(activation_function(z_values[j]));
                }
            }
                
            zs.push(z_values);
            current_input = outputs; // the outputs of the this layer will be the inputs for the next layer (repeat until all the last layer which is the output layer)
            all_layer_outputs.push(current_input); // Push the actual activations
        }
        // after all the layers gives off their outputs, return final array of current_input as the predictions
        return {
            predictions: current_input, 
            activations : all_layer_outputs,
            zs: zs
        };
    }
}

module.exports = Interpreter;
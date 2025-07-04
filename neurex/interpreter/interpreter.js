
/**
 * 

 This interpreter is dedicated for Neurex

 */


const activation = require('../activations');
const fs = require('fs');
const path = require('path');

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

     Thw loadSavedModel() method allows you to load the trained model. The model is typically in a JSON file format which contains the

     */
    loadSavedModel(model) {
        try {
            if (!model) {
                throw new Error("No model provided");
            }
            const dir = path.dirname(require.main.filename);
            const model_file = path.join(dir,`${model}`);
            const content = fs.readFileSync(model_file, 'utf-8');
            const params = JSON.parse(content);

            // mapping back the stored activation functions
            this.activation_functions = params.activation_functions.map(name => activation[name.toLowerCase()]);

            // storing the weights
            this.weights = params.weights;

            // storing the biases
            this.biases = params.biases;

            // getting the networks number of layers
            this.num_layers = params.num_layers;

            // get the neurons per layers. It is stored in array to easily map in what layer has this number of nuerons
            this.number_of_neurons = params.number_of_neurons;

            // get the input size
            this.input_size = params.input_size;

            console.log("Using:",model);
        }
        catch (error) {
            console.log(error);
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
                const {predictions} = this.#FeedForward(array_of_features);
                outputs.push(predictions[0]);
            }
            return outputs;
        }
        catch (err) {
            console.error(err.message);
        }
    }

    #FeedForward(input) {
        let current_input = input;

        /**
        
            Looping through the layers. The value of layer(int) will be use to index which biases, weights, and activations functions
            to use in this layer.
         */

        for (let layer = 0; layer.num_layers; layer++) {
            /**  get the array of biases from the array of arrays of biases 

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
             /**  get the array of weights from the array of arrays of weights 

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
             * 

            in the constructor, we have "this.number_of_neurons" which is an array. This is because when constructing the network using 
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
             * 
             in the cunstructor, we have "this.activation_functions". This is because when constructing the network using 
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
                        "relu", <- index 0 - this is the activation function to be use in this layer and will be use by all neurons for this layer
                        "relu", 
                        "linear", 
                        ....
                    ]
            */
            const activation_function = this.activation_functions[layer];
            let outputs = [];

            for (let neuron = 0; neuron < num_neurons; neuron++ ) {
                let dot_product_output = 0;

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
                dot_product_output += layer_biases[neuron];
                /**
                
                    Once we get the dot-product, we apply the activation function for this layer
                    Assume:
                        let layer = 0;

                    then:
                        activation_functions = [
                            "relu", <- index 0 - this is the activation function going to be use by all neurons in this layer
                            "relu", 
                            "linear", 
                            ....
                        ]
                 */
                outputs.push(activation_function(dot_product_output));
            }
            // the outputs of the this layer will be the inputs for the next layer (repeat until all the last layer which is the output layer)
            current_input = outputs; 
        }
        // after all the layers gives off their outputs, return final array of current_input as the predictions
        return {
            predictions : current_input
        };
    }
}

module.exports = Interpreter;
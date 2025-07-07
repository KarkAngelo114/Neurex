/**
 * Neurex - Feedforward Neural Network NodeJS library
 * Author: Kark Angelo V. Pada
 * 
 * Copyright (c) all rights reserved
 * 
 * Licensed under the MIT License.
 * See LICENSE file in the project root for full license information.
 * 
 */

/**
import necessary modules
 */

const fs = require('fs');
const path = require('path');
const activation = require('../activations');
const optimizers = require('../optimizers')
const lossFunctions = require('../loss_functions');


/**
 * Neurex is a configurable feedforward artificial neural network.
 * 
 * This class allows you to define the architecture of a neural network by specifying the number of layers,
 * neurons per layer, and activation functions. It supports training with various optimizers, saving
 * model state, and provides utility methods for inspecting the model structure.
 * 
 * @class
 * 
 * @property {Array<Array<Array<number>>>} weights - The weights for each layer, organized as 3D array [layer][input][output].
 * @property {Array<Array<number>>} biases - The biases for each layer, organized as 2D array [layer][neuron].
 * @property {number} learning_rate - The learning rate used during training.
 * @property {number} num_layers - The total number of layers in the network.
 * @property {Array<Function>} activation_functions - The activation functions for each layer.
 * @property {Array<Function>} derivative_functions - The derivatives of the activation functions for each layer.
 * @property {Array<number>} number_of_neurons - The number of neurons in each layer.
 * @property {number} input_size - The number of input features (input layer size).
 * @property {number} epoch_count - The number of epochs the model has been trained for.
 * @property {string} optimizer - The name of the optimizer used for training.
 * @property {Object} optimizerStates - Internal state for optimizers, storing per-layer weight and bias states.
 * 
 */

class Neurex {
    constructor () {
        this.weights = [];
        this.biases = [];
        this.num_layers = 0;
        this.activation_functions = [];
        this.derivative_functions = [];
        this.number_of_neurons = [];
        this.input_size = 0;
        this.accuracy = '';
        this.loss_function = '';
        this.output_size = 0;
        this.task = null;
        this.epoch_count = 0;
        this.batch_size = 0;

        // default configs
        this.optimizer = 'sgd';
        this.learning_rate = 0.001;
        this.randMin = -1;
        this.randMax = 1;

        // Optimizer state for each layer (weights and biases)
        this.optimizerStates = {
            weights: [], // Array of state objects for each layer's weights
            biases: []   // Array of state objects for each layer's biases
        };
    }

    /**
    * @typedef {Object} NeurexConfig
    * @property {number} [learning_rate] - Learning rate for training.
    * @property {string} [optimizer] - Optimizer to use [available: sgd, adam, adagrad, rmsprop, adadelta ].
    * @property {number} [randMin] - Minimum value for random initialization of weights/biases.
    * @property {number} [randMax] - Maximum value for random initialization of weights/biases.
    */

    /**
    * Allows configuration of your neural network's parameters.
    * @method configure
    * @param {NeurexConfig} configs - Configuration options for the neural network.
    *
    * You may configure them optionally. Be careful of tweaking them as they will have an effect on your model's performance.
    *
    * Default configurations:
    *   learning_rate: 0.001
    *   optimizer: 'adam'
    *   randMin: -1
    *   randMax: 1
    */
    configure(configs) {
        if (configs.learning_rate !== undefined) this.learning_rate = configs.learning_rate;
        if (configs.optimizer !== undefined) this.optimizer = configs.optimizer;
        if (configs.randMin !== undefined) this.randMin = configs.randMin;
        if (configs.randMax !== undefined) this.randMax = configs.randMax;
    }

    /**
     * @method inputShape()
     * @param {Array} data - the dataset

     the inputShape() method allows you to get the shape of your input.
     This will tell the network that your input layer has this X number of input neuron.
     Ensure that your dataset has no missing values, otherwise perform data cleaning.
     */

    inputShape(data) {
        // this refers to the first array of features and the number of features is the number of input neurons in the input layer
        // and assumes that the subsequent data has the same number of features
        // this is why it is important to perform data preprocessing at first before feeding to avoid having problems.
        this.input_size = data[0].length;
        let size = this.input_size;
        // do a loop to all the rows to check for shape inconsistencies
        try {
            data.forEach((rows, i) => {
                if (rows.length != size) {
                    throw new Error(`[ERROR]------- Shape mismatch on row${i}. Ensure that all shapes has the same shape.\nExpected shape: ${size}. Row${i} has ${rows.length}.`);
                }
            });
        }
        catch (error) {
            console.log(error);
        }
    }

    /**
     * 
     *
    @method flatten()  
    @param {Array} input - dataset
    @returns - a flattened array of dataset
    @throws if the required input is not present or it is not an array

    flattens an array of arrays (matrices) into 1D array
    Example:

    first row:
        [[x1, x2, x3], [x1, x2, x3], [x1, x2, x3], [x1, x2, x3], [x1, x2, x3], [x1, x2, x3]]

    flattened:

    [x1, x2, x3, x4, x5, x6, ....]
     */
    flatten(input) {
        let flattened_array = [];
        try {
            if (!Array.isArray(input)) {
                throw new Error("[ERROR]------- The input is not an array");
            }
            if (!input) {
                throw new Error("[ERROR]------- No input to flatten");
            }

            for (let i = 0; i < input.length; i++) {
                flattened_array.push(input[i].flat(Infinity));
            }
            
            return flattened_array;
        }
        catch (error) {
            console.error(error);
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
            const actName = actFn && actFn.name ? actFn.name : (typeof actFn === 'string' ? actFn : 'custom');
            console.log(`${i + 1}\t${this.number_of_neurons[i]}\t${actName}`);
        }
    }

    /**
     * 

     @method construct_layer()
     @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tahn, linear)
     @param {Number} layer_size specify the number of neuron for this layer.

     The construct_layer() method allows you to create layers of your network. 
     Each layer has its own number of neurons and all uses the same activation function
     to output new features before passing to the next layer.

     */

    construct_layer(activation_func, layer_size) {
        this.num_layers += 1;

        try {
            if (!activation_func || !layer_size || isNaN(layer_size)) {
                throw new Error(`There are/is missing parameter/s in layer ${this.num_layers}`);
            }

            const funcName = activation_func.toLowerCase();

            if (!activation[funcName] || !activation.derivatives[funcName]) {
                throw new Error(`Activation function '${funcName}' or its derivative not found`);
            }

            this.activation_functions.push(activation[funcName]);               // actual activation function
            this.derivative_functions.push(activation.derivatives[funcName]);  // actual derivative function
            this.number_of_neurons.push(layer_size);
        }
        catch (error) {
            console.error(error);
        }
    }

    /**
     * 
    @method saveModel()
    @param {string} modelName - the filename of your model

    saveModel() allows you to save your model's architecture, weights, and biases, as well as other parameters in a .json file. Once called after training,
    it will generate two .json files - the metadata.json and the actual model (YourModelName.json or if name not specified, it will be Model_date_today.json).
    
     */
    saveModel(modelName = null) {
        console.log("\n[TASK]------- Saving model's architecture...");
        let fileName = modelName;
        if (!modelName || modelName == null || modelName == undefined) {
            fileName = `Model_${new Date().toISOString().replace(/[:.]/g, '-')}`;
        }

        const data = {
            "task":this.task,
            "loss_function":this.loss_function,
            "epoch":this.epoch_count,
            "batch_size":this.batch_size,
            "optimizer":this.optimizer,
            "learning_rate":this.learning_rate,
            "activation_functions":this.activation_functions.map(functionName => functionName.name),
            "derivative_functions":this.derivative_functions.map(functionName => functionName.name),
            "input_size":this.input_size,
            "output_size":this.output_size,
            "num_layers":this.num_layers,
            "number_of_neurons":this.number_of_neurons,
            "weights":this.weights,
            "biases":this.biases,
            
        };

        const metadata = [
            this.epoch_count,
            this.optimizer,
            this.loss_function,
            this.task
        ];
        this.#save(data, fileName, metadata);
        
    }

    /**
    * Trains the neural network using the provided training data, target values, number of epochs, and learning rate.
    * 
    * This method initializes the weights and biases for each layer, then iteratively performs forward propagation,
    * computes the loss, backpropagates the error, and updates the weights and biases using gradient descent.
    *
    * @method train()
    * @param {Array<Array<number>>} trainX - The input training data. Each element is an array representing a single sample's features.
    * @param {Array<number>} trainY - The target values (ground truth) corresponding to each sample in trainX.
    * @param {string} loss - loss function to use (Accuracy: MSE, MAE, binary_crossentropy, categorical_crossentropy)
    * @param {Number} epoch - the number of training iteration
    * @param {Number} batch_size - mini batch sizing
    * 
    * @throws {Error} Throws an error if any required parameter is missing.
    * @returns Progress of every epoch can be print in the console.
    * 
    * @example
    * // Example usage:
    * const nn = new core1();
    * nn.inputShape(trainX); // Set input shape based on your data
    * nn.construct_layer('relu', 8); // Add a hidden layer with 8 neurons and ReLU activation
    * nn.construct_layer('linear', 1); // Add an output layer with 1 neuron and linear activation
    * nn.train(trainX, trainY, 'mse', 100, 4); // Train for 100 epochs with learning rate 0.01 a loss function of 'mse' and a batch size of 4
    * 
    * After training, you can use the network for predictions
    */

    train(trainX, trainY, loss, epoch, batch_size) { // since this is core 1, this will focus on single output regression task
        // initialize biases
        let prev_size = this.input_size;
        for (let i = 0; i < this.num_layers; i++) {
            let generated_biases = [];
            for (let j = 0; j < this.number_of_neurons[i]; j++) {
                generated_biases.push(Math.random() * (this.randMax - this.randMin) + this.randMin);
            }
            this.biases.push(generated_biases);
        }

        // initialize weights
        for (let i = 0; i < this.number_of_neurons.length; i++) {
            let layer_size = this.number_of_neurons[i];
            let layerWeights = [];
            for (let r = 0; r < prev_size; r++) {
                let row = [];
                for (let c = 0; c < layer_size; c++) {
                    row.push(Math.random() * (this.randMax - this.randMin) + this.randMin);
                }
                layerWeights.push(row);
            }
            this.weights.push(layerWeights);
            prev_size = layer_size;
        }

        /**
         * 
         get the output size of neuron

         assume: 
            this.number_of_neurons = [6, 7, 1];
            index 0 (layer 1) = 6
            index 1 (layer 2) = 7
            index 2 (layer 2) = 1

            const number_of_neurons = [6, 7, 1];
            const last_value = number_of_neurons[number_of_neurons.length - 1];
            console.log(last_value); // which is 1
         */
        this.output_size = this.number_of_neurons[this.number_of_neurons.length - 1];

        // Initialize optimizer state for each layer
        this.optimizerStates = {
            weights: Array(this.num_layers).fill().map(() => []),
            biases: Array(this.num_layers).fill().map(() => [])
        };

        try {
            if (!trainX || !trainY || !loss) {
                throw new Error(`[FAILED]------- There is/are missing parameter/s. Failed to start training...`);
            }

            if (epoch == 0 || batch_size == 0 || !epoch || !batch_size) {
                throw new Error("[FAILED]------- Epoch or batch size cannot be zero");
            }

            this.loss_function = loss.toLowerCase();
            const loss_function = lossFunctions[this.loss_function.toLowerCase()];
            const optimizerFn = optimizers[this.optimizer.toLowerCase()];
            
            this.epoch_count = epoch;
            this.batch_size = batch_size;
            const batchSize = batch_size;
            
            
            // Infer task type based on output layer and loss/activation
            const lastLayerActivation = this.activation_functions[this.activation_functions.length - 1].name;
            const lossLower = loss.toLowerCase();
            
            // Regression: output_size == 1, activation linear, loss mse/mae
            if (this.output_size == 1 && lastLayerActivation === 'linear' && (lossLower === 'mse' || lossLower === 'mae')) {
                this.task = 'regression';
            }
            else {
                throw new Error(`[ERROR]------- Using ${lossLower} having output size of ${this.output_size} and a ${lastLayerActivation} function in the output layer is currently unavailable for this core's task.`);
            }

            if (!optimizerFn) throw new Error(`Unknown optimizer: ${this.optimizer}`);
            console.log("\n[TASK]------- Training session is starting\n");

            // epoch loop
            for (let current_epoch = 0; current_epoch < epoch; current_epoch++) {
                let totalepochLoss = 0;

                // batch size
                for (let batchStart = 0; batchStart < trainX.length; batchStart += batchSize) {

                    const batchEnd = Math.min(batchStart + batchSize, trainX.length);
                    const actualBatchSize = batchEnd - batchStart;

                    // Initialize accumulators for gradients
                    let weightGrads = this.weights.map(layer => layer.map(row => row.map(() => 0)));
                    let biasGrads = this.biases.map(layer => layer.map(() => 0));

                    let batchLoss = 0;

                    // Accumulate gradients for each sample in the batch
                    for (let sample_index = batchStart; sample_index < batchEnd; sample_index++) {

                        let input = trainX[sample_index];
                        let actual = trainY[sample_index];

                        // feed forward
                        const {predictions, activations, zs} = this.#Feedforward(input);
                        const deltas = [];


                        // === STEP 1: Compute delta for output layer === //
                        let output_layer = this.num_layers - 1;
                        let dOutputlayer = [];

                        batchLoss += loss_function(predictions, actual);

                        
                        for (let j = 0; j < this.number_of_neurons[output_layer]; j++) {
                            const error = predictions[j] - actual;
                            const dAct = this.derivative_functions[output_layer](zs[output_layer][j]);
                            dOutputlayer.push(error * dAct);
                        }

                        deltas[output_layer] = dOutputlayer;
                        
                        // backpropagate to the hidden layers (except input layer)

                        for (let layer = this.num_layers - 2; layer >= 0; layer--) {
                            let current_delta = [];
                            const weights_next = this.weights[layer+1];
                            const next_delta = deltas[layer+1];

                            for (let neuron = 0; neuron < this.number_of_neurons[layer]; neuron++) {
                                let sum = 0;
                                for (let i = 0; i < this.number_of_neurons[layer+1]; i++) {
                                    sum += weights_next[neuron][i] * next_delta[i];
                                }
                                const derivative_activation = this.derivative_functions[layer](zs[layer][neuron]);

                                current_delta.push(sum * derivative_activation);
                            }
                            deltas[layer] = current_delta;
                        }

                        // === STEP 3: Accumulate Gradients === //
                        for (let l = 0; l < this.num_layers; l++) {
                            const delta = deltas[l];
                            const a_prev = activations[l];
                            // Accumulate weight gradients
                            for (let i = 0; i < this.weights[l].length; i++) {
                                for (let j = 0; j < this.weights[l][i].length; j++) {
                                    weightGrads[l][i][j] += a_prev[i] * delta[j];
                                }
                            }
                            // Accumulate bias gradients
                            for (let j = 0; j < this.biases[l].length; j++) {
                                biasGrads[l][j] += delta[j];
                            }
                        }
                    }

                    batchLoss /= actualBatchSize;
                    totalepochLoss += batchLoss;

                    // Divide accumulated gradients by the actual batch size
                    for (let l = 0; l < this.num_layers; l++) {
                        for (let i = 0; i < weightGrads[l].length; i++) {
                            for (let j = 0; j < weightGrads[l][i].length; j++) {
                                weightGrads[l][i][j] /= actualBatchSize;
                            }
                        }
                        for (let i = 0; i < biasGrads[l].length; i++) {
                            biasGrads[l][i] /= actualBatchSize;
                        }
                    }

                    for (let l = 0; l < this.num_layers; l++) {
                        // Update weights
                        this.optimizerStates.weights[l] = optimizerFn(
                            this.weights[l],
                            weightGrads[l],
                            this.optimizerStates.weights[l],
                            this.learning_rate
                        );
                        // Update biases
                        this.optimizerStates.biases[l] = optimizerFn(
                            this.biases[l],
                            biasGrads[l],
                            this.optimizerStates.biases[l],
                            this.learning_rate
                        );
                    }

                }

            let AverageEpochLoss = totalepochLoss / batchSize;
            console.log(`Epoch ${current_epoch+1}/${epoch} | Loss: ${AverageEpochLoss.toFixed(5)}`);
            }
            

        }
        catch (error) {
            console.log(error);
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
                let input_data = input[sample_index];

                const {predictions} = this.#Feedforward(input_data);
                outputs.push(predictions[0]);
            }

            return outputs;
        }
        catch (error) {
            console.log(error);
        }
    }

    // ========= Private methods =======
    // forward propagation
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
                dot_product_output += layer_biases[neuron]; // adds the bias for this neuron
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
                z_values.push(dot_product_output);
                if (activation_function.name === "softmax") {
                    outputs = activation_function(z_values);
                }
                else {
                    outputs.push(activation_function(dot_product_output)); // apply the activation function that is assigned for the layer where this neuron is and then push the results in the array of outputs;
                }
                
            }
            zs.push(z_values);
            current_input = outputs; // the outputs of the this layer will be the inputs for the next layer (repeat until all the last layer which is the output layer)
            all_layer_outputs.push(outputs);
        }
        // after all the layers gives off their outputs, return final array of current_input as the predictions
        return {
            predictions: current_input, 
            activations : all_layer_outputs,
            zs: zs
        };
    }

    //saving model
    #save(data, fileName, meta) {
        const dir = path.dirname(require.main.filename);
        const metadata = {
            "Date Created":`${new Date().toISOString().replace(/[:.]/g, '-')}}`,
            "Number of epoch to train":meta[0],
            "Optimizer":meta[1],
            "Loss function":meta[2],
            "Task":meta[3],
            "Trained using":"Neurex",
            "Note":"This model can only be use on Neurex library. This cannot be use directly to other known machine-learning frameworks/libraries. DO NOT modify any of the parameters."
        }

        const path_to_file = path.join(dir,`${fileName}.json`);
        const path_to_file_metadata = path.join(dir,`metadata.json`);
        const content = JSON.stringify(data, null, 2);
        const metadataContent = JSON.stringify(metadata, null, 2);
        fs.writeFileSync(path_to_file, content);
        fs.writeFileSync(path_to_file_metadata, metadataContent);

        console.log("[SUCCESS]------- Model is save as "+fileName+".json");
    }
}

module.exports = Neurex;
/**
 * Neurex - Feedforward Neural Network NodeJS library
 * Author: Kark Angelo V. Pada
 * * Copyright (c) all rights reserved
 * * Licensed under the MIT License.
 * See LICENSE file in the project root for full license information.
 * */

/**
import necessary modules
 */

const fs = require('fs');
const zlib = require('zlib');
const path = require('path');
const { scaleGradients} = require('../internals/gradientKernels');
const optimizers = require('../optimizers')
const lossFunctions = require('../loss_functions');
const color = require('../prettify');
const { computeWeightGradients } = require('../core/bindings');
const { getShape, flattenAll, calculateTensorShape } = require('../utils');



/**
 * Neurex is a configurable feedforward artificial neural network.
 * * This class allows you to define the architecture of a neural network by specifying the number of layers,
 * neurons per layer, and activation functions. It supports training with various optimizers, saving
 * model state, and provides utility methods for inspecting the model structure.
 * @class
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
 * */

class Neurex {
    constructor () {
        this.weights = [];
        this.biases = [];
        this.num_layers = 0;
        this.input_size = 0;
        this.input_shape = [];
        this.accuracy = '';
        this.loss_function = '';
        this.output_size = 0;
        this.task = null;
        this.epoch_count = 0;
        this.batch_size = 0;
        this.depth = 0;
        this.filters = 1;
        this.layers = []; // layers (except input type layers) and their details will store here
        this.hasSequentiallyBuild = false;
        this.hasBuilt = false;

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

        this.onGPU = false;
        this.isfailed = false;
        this.paramRegistry = [];
        this.totalParams = 0;
        this.weightGrads = [];
        this.biasGrads = [];
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
    * learning_rate: 0.001
    * optimizer: 'adam'
    * randMin: -1
    * randMax: 1
    */
    configure(configs) {
        if (configs.learning_rate !== undefined) this.learning_rate = configs.learning_rate;
        if (configs.optimizer !== undefined) this.optimizer = configs.optimizer;
        if (configs.randMin !== undefined) this.randMin = configs.randMin;
        if (configs.randMax !== undefined) this.randMax = configs.randMax;
    }

    /**
     * @method modelSummary()

    Shows the model architecture
     */
    /**
     * @method modelSummary()
     * Shows the model architecture
     */
    modelSummary() {
        console.log("_______________________________________________________________");
        console.log("                        Model Summary                          ");
        console.log("_______________________________________________________________");
        console.log(`Input size: ${this.input_size}`);
        console.log(`Number of layers: ${this.num_layers}`);
        console.log("---------------------------------------------------------------");
        console.log("Layer (type)              Output Shape          Activation     ");
        console.log("===============================================================");
        
        // Initialize currentShape with the input shape
        let currentShape = [...this.input_shape];
        let [h, w, d] = currentShape;

        this.layers.forEach((layer, i) => {
            const layerType = layer.layer_name;
            const activationName = layer.activation_function ? layer.activation_function.name : 'None';

            if (layerType === 'convolutional2D') {
                const [kernelH, kernelW] = layer.kernel_size;
                const stride = layer.strides;
                const padding = layer.padding.toLowerCase();
                const filters = layer.filters;
                if (padding === 'same') {
                    // Same padding preserves spatial size, so h and w remain unchanged
                } else if (padding === 'valid') {
                    h = Math.floor((h - kernelH) / stride + 1);
                    w = Math.floor((w - kernelW) / stride + 1);
                }

                const newShape = [h, w, filters];
                const warning = (h <= 0 || w <= 0) ? '⚠️ INVALID OUTPUT SHAPE' : '';

                console.log(
                    `Convolutional layer      (${newShape.join('x')})        ${activationName.padEnd(10)} ${warning}`
                );

                // Update currentShape for the next layer
                currentShape = newShape;
                [h, w, d] = currentShape;
            } else if (layerType === 'connected_layer') {
                const outputShape = layer.layer_size;
                console.log(
                    `Connected Layer          (1x1x${outputShape})           ${activationName.padEnd(10)}`
                );
                currentShape = [1, 1, outputShape]; // assume flat
                [h, w, d] = currentShape;
            } else if (layerType === 'flatten_layer') {
                const size = h * w * d;
                currentShape = [1, 1, size];
                console.log(
                    `Flatten Layer           (1x1x${size})        None`
                );
                [h, w, d] = currentShape;
            }
        });
        const total_weights = this.weights.flat(Infinity).length
        const total_biases = this.biases.flat(Infinity).length
        console.log("===============================================================");
        console.log("Total layers: " + this.num_layers);
        console.log("Total Learnable parameters:",parseInt(total_weights+total_biases));
        console.log("===============================================================");

        if (this.weightGrads.flat(Infinity).some(isNaN)) throw new Error('There are NaNs on weight grads');
        
    }

    /**
    * 
     @method saveModel()
     @param {string} modelName - the filename of your model

     saveModel() allows you to save your model's architecture, weights, and biases, as well as other parameters. The model will be exported
     as a .nrx (neurex) model and a metadata.json will be generated along with the model file.
        
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
            "layers": this.layers.map(layer => ({
                layer_name: layer.layer_name,
                activation_function_name: layer.activation_function ? layer.activation_function.name : null,
                derivative_activation_function_name: layer.derivative_activation_function ? layer.derivative_activation_function.name : null,
                layer_size: layer.layer_size || null,
                feedforward: layer.feedforward,
                backpropagate: layer.backpropagate
            })),
            "input_size":this.input_size,
            "output_size":this.output_size,
            "num_layers":this.num_layers,
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
     * 
     * @method sequentialBuild
     * 
     * interface to stack layer types. No weights and biases initialization here
     * @param {Object} layer_data
     */
    sequentialBuild(layer_data) {

        try {

            if (!layer_data || layer_data.length < 2) {
                throw new Error("[ERROR]------- No layers");
            }

            layer_data.forEach(layer => {
                // extract input size
                if (layer.layer_name === "input_layer") {
                    this.input_size = layer.layer_size;
                    this.input_shape = layer.input_shape || 0;
                    this.depth = this.input_shape[2] || 0;
                }
                else {
                    this.layers.push(layer);
                }
            });

            this.hasSequentiallyBuild = true;
            this.num_layers = this.layers.length;
            this.#build();
            return layer_data; 
        }
        catch(err) {
            console.error(err);
        }
        

    }

    /**
    * Trains the neural network using the provided training data, target values, number of epochs, and learning rate.
    * * This method initializes the weights and biases for each layer, then iteratively performs forward propagation,
    * computes the loss, backpropagates the error, and updates the weights and biases using gradient descent.
    *
    * @method train()
    * @param {Array<Array<number>>} trainX - The input training data. Each element is an array representing a single sample's features.
    * @param {Array<number>} trainY - The target values (ground truth) corresponding to each sample in trainX.
    * @param {string} loss - loss function to use: MSE, MAE, binary_crossentropy, categorical_crossentropy, sparse_categorical_cross_entropy
    * @param {Number} epoch - the number of training iteration
    * @param {Number} batch_size - mini batch sizing
    * @throws {Error} Throws an error if any required parameter is missing.
    * @returns Progress of every epoch can be print in the console.
    * * @example
    * // Example usage:
        * 
        * const {Neurex, Layers} = require('neurex');
        * const model = new Neurex();
        * const layer = new Layers();
        *
        * model.sequentialBuild([
        *    layer.inputShape(X_train),
        *    layer.connectedLayer("relu", 3),
        *    layer.connectedLayer("relu", 3),
        *    layer.connectedLayer("softmax", 2)
        * ]);
        * model.build();
        *
        * model.train(X_train, Y_train, 'categorical_cross_entropy', 2000, 12);
    * * After training, you can use the network for predictions
    */

    train(trainX, trainY, loss, epoch, batch_size) {

        const lastLayerObject = this.layers[this.layers.length - 1];
        this.output_size = lastLayerObject.layer_size;

        // Initialize optimizer state for each layer
        this.optimizerStates = {
            weights: Array(this.num_layers).fill().map(() => ({})),
            biases: Array(this.num_layers).fill().map(() => ({}))
        };

        try {
            if (!trainX || !trainY || !loss) {
                this.isfailed = true;
                throw new Error(`[FAILED]------- There is/are missing parameter/s. Failed to start training...`);
            }

            if (epoch == 0 || batch_size == 0 || !epoch || !batch_size) {
                this.isfailed = true;
                throw new Error("[FAILED]------- Epoch or batch size cannot be zero");
            }

            this.loss_function = loss.toLowerCase();
            const loss_function = lossFunctions[this.loss_function.toLowerCase()];
            const optimizerFn = optimizers[this.optimizer.toLowerCase()];
            
            this.epoch_count = epoch;
            this.batch_size = batch_size;
            const batchSize = batch_size;
            
            
            // Infer task type based on output layer and loss/activation
            const lastLayerActivation = lastLayerObject.activation_function.name;
            const lossLower = loss.toLowerCase();
            
            // Regression: output_size == X, activation linear, loss mse/mae
            if (lastLayerActivation === 'linear' && (lossLower === 'mse' || lossLower === 'mae')) {
                this.task = 'regression';
            }
            // binary classification task: activation in output layer = sigmoid, loss = binary_cross_entropy
            else if (lastLayerActivation === "sigmoid" && lossLower === 'binary_cross_entropy') {
                this.task = 'binary_classification';
            }
            // multi-class classification task: activation in output layer = softmax, loss = categorical_cross_entropy (labels must be one-hot encoding)
            else if (lastLayerActivation === 'softmax' && lossLower === 'categorical_cross_entropy') {
                // do a loop if any of the rows of trainY is not
                trainY.forEach(row => {
                    if (this.output_size != row.length) {
                        this.isfailed = true;
                        throw new Error(`[ERROR]------- Output shape mismatch. The size of the output layer must be the same number of classes`);
                    }
                });

                // check if the Y_train is not one-hot encoded
                const isOneHotEncoded = this.#ifOneHotEndcoded(trainY);
                if (isOneHotEncoded) {
                    this.task = 'multi_class_classification';
                }
                else {
                    this.isfailed = true;
                    throw new Error("[ERROR]------- Y_train must be one-hot encoded for the categorical_cross_entropy loss. Use 'sparse_categorical_cross_entropy' instead if the Y_train is interger-encoded labels.");
                }
                
            }
            else if (lastLayerActivation === 'softmax' && lossLower === 'sparse_categorical_cross_entropy') {
                this.task = 'multi_class_classification';
            }
            else {
                this.isfailed = true;
                throw new Error(`[ERROR]------- Using ${lossLower} having output size of ${this.output_size} and a ${lastLayerActivation} function in the output layer is currently unavailable for this core's task.`);
            }

            if (!optimizerFn) {
                this.isfailed = true;
                throw new Error(`Unknown optimizer: ${this.optimizer}`)
            };
            console.log("\n[TASK]------- Training session is starting\n");

            // epoch loop
            for (let current_epoch = 0; current_epoch < epoch; current_epoch++) {
                let totalepochLoss = 0;
                let numBatches = 0; // Added to count batches

                // batch size
                for (let batchStart = 0; batchStart < trainX.length; batchStart += batchSize) {
                    numBatches++; // Increment batch count

                    const batchEnd = Math.min(batchStart + batchSize, trainX.length);
                    const actualBatchSize = batchEnd - batchStart;

                    // Initialize accumulators for gradients
                    // function zerosLike(arr) {
                    //     if (Array.isArray(arr)) {
                    //         return arr.map(zerosLike);
                    //     }
                    //     return 0;
                    // }

                    // let weightGrads = this.weights.map(zerosLike);
                    // let biasGrads = this.biases.map(zerosLike);
                    
                    // let weightGrads = this.weights.map(layer => layer.map(row => row.map(() => 0)));
                    // let biasGrads = this.biases.map(layer => layer.map(() => 0));

                    let weightGrads = this.weightGrads;

                    let biasGrads = this.biasGrads;


                    let batchLoss = 0;

                    // Accumulate gradients for each sample in the batch
                    for (let sample_index = batchStart; sample_index < batchEnd; sample_index++) {

                        let input = trainX[sample_index];
                        let actual = trainY[sample_index];

                        // feed forward
                        const {predictions, activations, zs} = this.#Feedforward(input);
                        const deltas = [];


                        // === STEP 1: Compute delta for output layer === //
                        let output_layer_index = this.num_layers - 1;
                        let dOutputlayer = [];
                        const network_output_layer = this.layers[output_layer_index];

                        batchLoss += loss_function(predictions, actual);

                        
                        for (let j = 0; j < network_output_layer.layer_size; j++) {
                            if (this.task === "binary_classification") {
                                // binary classification
                                dOutputlayer.push(predictions[j] - actual[j]);
                            }
                            else if (this.task === "multi_class_classification") {
                                if (lastLayerActivation === 'softmax' && lossLower === "categorical_cross_entropy") {
                                    dOutputlayer.push(predictions[j] - actual[j]);
                                }
                                else if (lastLayerActivation === 'softmax' && lossLower === "sparse_categorical_cross_entropy") {
                                    dOutputlayer = [...predictions];
                                    dOutputlayer[actual[0]] -= 1; 
                                }
                                else {
                                    throw new Error(`[ERROR]------- Uknown loss function for multi-class classification loss. Loss: ${lossLower} is unknown.`)
                                }
                            }
                            else {
                                // regression tasks single or multi-output regression
                                const error = predictions[j] - actual[j];
                                //const dAct = this.derivative_functions[output_layer](zs[output_layer][j]);
                                const dAct = network_output_layer.derivative_activation_function([zs[output_layer_index][j]]);
                                dOutputlayer.push(error * dAct);
                            }
                        }

                        deltas[output_layer_index] = dOutputlayer;

                        // backpropagation loop
                        const allDeltas = this.#backpropagation(activations, zs, deltas);

                        // === STEP 3: Accumulate Gradients === //
                        for (let l = 0; l < this.num_layers; l++) {
                            const delta = allDeltas[l];
                            const a_prev = activations[l];

                            if (weightGrads[l].flat(Infinity).some(isNaN)) {
                                console.log(`Weight grads for layer ${l+1}`,weightGrads[l][0]);

                                this.weightGrads[l].flat(Infinity).forEach(w => console.log(w))
                                throw new Error(`Error occured when computing weight gradients on layer ${l+1}: "${this.layers[l].layer_name}". Reason: "weightGrads" has NaNs. If this error occured, please report this error`); 
                            }

                            throw new Error();

                            // console.log(`Delta for layer ${l}`,delta);
                            // console.log(`Activated outputs for layer ${l}`, a_prev);
                            // console.log(`Weight grads for layer ${l}`,weightGrads[l]);

                            // compute for weight gradients
                            weightGrads[l] = computeWeightGradients(a_prev, delta, this.layers[l].layer_name, weightGrads[l], this.layers[l], allDeltas, l);

                            // Accumulate bias gradients
                            for (let j = 0; j < biasGrads[l].length; j++) {
                                biasGrads[l][j] += delta[j];
                            }
                        }
                    }

                    batchLoss /= actualBatchSize;
                    totalepochLoss += batchLoss;

                    const flattenWeightGrads = flattenAll(this.weightGrads);
                    const flattenWeights = flattenAll(this.weights);

                    // throw new Error('Stopping');

                    // Divide accumulated gradients by the actual batch size
                    for (let l = 0; l < this.num_layers; l++) {

                        if (this.layers[l].layer_name === "connected_layer") {

                            for (let i = 0; i < weightGrads[l].length; i++) {
                                weightGrads[l][i] = scaleGradients(this.onGPU, weightGrads[l][i], actualBatchSize);
                            }
                            biasGrads[l] = scaleGradients(this.onGPU, biasGrads[l], actualBatchSize);
                        }
                        
                    }


                    for (let l = 0; l < this.num_layers; l++) {
                        // Update weights

                        if (this.layers[l].layer_name === "connected_layer") {
                            this.optimizerStates.weights[l] = optimizerFn(
                                this.onGPU,
                                this.weights[l],
                                weightGrads[l],
                                this.optimizerStates.weights[l],
                                this.learning_rate
                            );
                            // Update biases
                            this.optimizerStates.biases[l] = optimizerFn(
                                this.onGPU,
                                this.biases[l],
                                biasGrads[l],
                                this.optimizerStates.biases[l],
                                this.learning_rate
                            );
                        }
                        
                    }

                }

                let AverageEpochLoss = totalepochLoss / numBatches; 
                let setColor = AverageEpochLoss > 0.09 ? color.red : 
                                AverageEpochLoss > 0.06 ? color.orange :
                                AverageEpochLoss > 0.04 ? color.yellow :
                                AverageEpochLoss > 0.03 ? color.lime : color.green;
                
                let logMessage = `[Epoch] ${current_epoch+1}/${epoch} | [Loss]: ${setColor} ${AverageEpochLoss.toFixed(7)} ${color.reset}`;

                if (this.task === 'binary_classification' || this.task === 'multi_class_classification') {
                    let epochPredictions = [];
                    for (let i = 0; i < trainX.length; i++) {
                        epochPredictions.push(this.#Feedforward(trainX[i]).predictions);
                    }
                    const accuracy = this.#calculateClassificationAccuracy(epochPredictions, trainY, this.task);

                    let accuracyColor = accuracy > 90 ? color.green :
                                    accuracy > 85 ? color.lime :
                                    accuracy >= 75 ? color.yellow :
                                    accuracy >= 60 ? color.orange : color.red;

                    logMessage += ` | [Accuracy in Training]: ${accuracyColor} ${accuracy.toFixed(2)}% ${color.reset}`;
                }
                console.log(logMessage);
            }
            
        }
        catch (error) {
            console.log(error);
        }
    }

    /**
     *  @method predict()
        @param {Array} input - input data 
        @returns Array of predictions
        @throws Error when there's shape mismatch and no input data

     produces predictions based on the input data
    */
    predict(input) {
        this.onGPU = false;
        try {
            if (!input) {
                throw new Error("\n[ERROR]-------No inputs")
            }

            let inputTensor = input[0][0];

            // if (Array.isArray(inputTensor)) {
            //     const size = this.input_shape[0]*this.input_shape[1]*this.input_shape[2];
            //     const height = inputTensor.length;
            //     const width = inputTensor[0].length;
            //     const depth = inputTensor[0][0].length;

            //     if (size != (height*width*depth)) {
            //         throw new Error(`[ERROR]------- Shape Mismatch | Input shape: (${height}, ${width}, ${depth}) | Expecting: (${this.input_shape[0]}, ${this.input_shape[1]}, ${this.input_shape[2]})`);
            //     }
            // }
            // else {
            //     if (input[0].length != this.input_size) {
            //         throw new Error(`\n[ERROR]-------Shape Mismatch | Input shape length: ${input[0].length} | Expecting ${this.input_size}`);
            //     }
            // }
            

            let outputs = [];
            for (let sample_index = 0; sample_index < input.length; sample_index++) {
                let input_data = input[sample_index];

                const {predictions} = this.#Feedforward(input_data);
                outputs.push(predictions);
            }

            return outputs;
        }
        catch (error) {
            console.log(error);
        }
    }

    // ========= Private methods =======

    #build() {
        try {

            let prev_size = this.input_size;
            let prevDepth = this.depth; // input layer depth
            let offset = 0;

            let input_Height = this.input_shape[0] || 0;
            let input_Width = this.input_shape[1] || 0;

            // initialized weights and biases
            this.layers.forEach((layer_data, idx) => {
                if (layer_data.layer_name === "connected_layer") {
                    let layer_size = layer_data.layer_size;

                    // initialize biases
                    let generated_biases = [];
                    for (let j = 0; j < layer_size; j++) {
                        generated_biases.push(Math.random() * (this.randMax - this.randMin) + this.randMin);
                    }
                    this.biases.push(generated_biases);

                    // initialized bias grads;
                    let initiatedBiasGrads = [];
                    for (let j = 0; j < layer_size; j++) {
                        initiatedBiasGrads.push(0);
                    }
                    this.biasGrads.push(initiatedBiasGrads);

                    // initialize weights
                    let layerWeights = [];
                    for (let r = 0; r < prev_size * this.filters; r++) {
                        let row = [];
                        for (let c = 0; c < layer_size; c++) {
                            row.push(Math.random() * (this.randMax - this.randMin) + this.randMin);
                        }
                        layerWeights.push(row);
                    }
                    this.weights.push(layerWeights);

                    // initialized weight grads
                    let initiated_weightGrads = [];
                    for (let r = 0; r < prev_size * this.filters; r++) {
                        let row = [];
                        for (let c = 0; c < layer_size; c++) {
                            row.push(0);
                        }
                        initiated_weightGrads.push(row);
                    }
                    this.weightGrads.push(initiated_weightGrads);


                    this.filters = 1; // we reset it to 1 so that the value of this.filters is only used by the first connected layer only after flatten
                    prev_size = layer_size;
                }
                else if (layer_data.layer_name === "convolutional2D") {
                    const filters = layer_data.filters;
                    const [kernelHeight, kernelWidth] = layer_data.kernel_size;
                    const stride = layer_data.strides || 1;
                    const padding = layer_data.padding || "same";

                    const depth = prevDepth;


                    // initialize kernels
                    let kernels = [];
                    for (let numFilters = 0; numFilters < filters; numFilters++) {
                        let rows = [];
                        for (let height = 0; height < kernelHeight; height++) {
                            let row_element = [];
                            for (let width = 0; width < kernelWidth; width++) {
                                let depth_elements = [];
                                for (let num_depth_elements = 0; num_depth_elements < depth; num_depth_elements++) {
                                    depth_elements.push(Math.random() * (this.randMax - this.randMin)+this.randMin);
                                }
                                row_element.push(depth_elements);
                            }
                            rows.push(row_element);
                        }
                        kernels.push(rows);
                    }
                    this.weights.push(kernels);

                    // initialize kernels grads
                    // let kernelsGrads = [];
                    // for (let numFilters = 0; numFilters < filters; numFilters++) {
                    //     let grad_rows = [];
                    //     for (let height = 0; height < kernelHeight; height++) {
                    //         let grad_row_element = [];
                    //         for (let width = 0; width < kernelWidth; width++) {
                    //             let grad_depth_elements = [];
                    //             for (let num_depth_elements = 0; num_depth_elements < depth; num_depth_elements++) {
                    //                 grad_depth_elements.push(0);
                    //             }
                    //             grad_row_element.push(grad_depth_elements);
                    //         }
                    //         grad_rows.push(grad_row_element);
                    //     }
                    //     kernelsGrads.push(grad_rows);
                    // }
                    // this.weightGrads.push(kernelsGrads);

                    let kernelGrads = Array.from({length: filters}, 
                        () => Array.from({length: kernelHeight},
                            () => Array.from({length: kernelWidth}, 
                                () => Array.from({length: depth}).fill(0)
                            )
                        )
                    )

                    this.weightGrads.push(kernelGrads)

                    // initialize bias for each kernel
                    let biases = [];
                    for (let numFilters = 0; numFilters < filters; numFilters++) {
                        biases.push(Math.random() * (this.randMax - this.randMin)+this.randMin);
                    }
                    this.biases.push(biases);

                    // initialized bias grads for kernels
                    let biasesKernelGrads = [];
                    for (let numFilters = 0; numFilters < filters; numFilters++) {
                        biasesKernelGrads.push(0);
                    }
                    this.biasGrads.push(biasesKernelGrads);
                    prevDepth = filters;
                    
                    const {OutputHeight, OutputWidth, CalculatedTensorShape} = calculateTensorShape(input_Height, input_Width, kernelHeight, kernelWidth, prevDepth, stride, padding);
                    input_Height = OutputHeight;
                    input_Width = OutputWidth;
                    prev_size = CalculatedTensorShape;
                    
                }
                // else if (layer_data.layer_name === "flatten_layer") {
                //     // flatten layer supposedly has no weights and biases, but in order to mitigate mismatching, we initialized weights and biases but all are 0s;
                //     let weights = Array(prev_size).fill(0);
                //     let biases = Array(prev_size).fill(0);
                //     this.weights.push(weights);
                //     this.biases.push(biases);
                //     this.weightGrads.push(weights);
                //     this.biasGrads.push(biases);
                //     this.filters = this.layers[idx-1].filters;
                //     prev_size = this.input_shape[0] * this.input_shape[1] * this.input_shape[2];
                // }
                // else if (layer_data.layer_name === "flatten_layer") {
                //     // flatten has no weights
                //     // but it changes the dimensionality
                //     prev_size = this.input_shape[0] * this.input_shape[1] * this.input_shape[2];
                //     this.filters = 1;
                // }

            });

            this.hasBuilt = true;

            function registerParam(tensor, meta) {
                const flatSize = tensor.flat(Infinity).length;
                this.paramRegistry.push({
                    offset,
                    size: flatSize,
                    shape: getShape(tensor),
                    ...meta
                });
                offset += flatSize;
            }

            // register weights
            this.weights.forEach((W, l) => {
                if (this.layers[l].layer_name === 'flatten_layer') return;
                registerParam.call(this, W, { layer: l, type: 'weights' });
            });

            // register biases
            this.biases.forEach((B, l) => {
                if (this.layers[l].layer_name === 'flatten_layer') return;
                registerParam.call(this, B, { layer: l, type: 'biases' });
            });

            this.totalParams = offset;

        }
        catch (error) {
            console.error(error);
        }
    }

    // backpropagation
    #backpropagation(activations, zs, deltas_array) {
        // The loop should iterate from the second-to-last layer (this.num_layers - 2) down to the first layer (0).

        let deltas = deltas_array;
        let delta_indexer = this.num_layers - 2;
        for (let layer_index = delta_indexer; layer_index >= 0; layer_index--) {
            // inline condition to check what layer need to get it's weights.
            // when initializing weights, flatten layer has initialized weights and biases, but all are 0s. So we neeed to skip it because if we
            // don't, the convolutional layer will get wrong kernels (weights)
            
            //const next_weights = this.weights[this.layers[layer_index].layer_name === "convolutional2D" ? delta_indexer : delta_indexer+1];
            
            const next_weights = this.weights[layer_index + 1]
            const next_delta = deltas[layer_index + 1];
        

            if (!Array.isArray(next_delta)) {
                throw new Error(`deltaNext at layer ${layer_index + 1} is undefined`);
            }
            
            // Get the current layer object, which now holds its backpropagation logic
            const currentLayer = this.layers[layer_index];

            // Call the layer's specific backpropagation method
            const {current_delta, decrementor_value} = currentLayer.backpropagate(this.onGPU,next_weights, next_delta, zs, layer_index, currentLayer, this.weights, activations);
            delta_indexer--;
            
            deltas[layer_index] = current_delta;
        }

        return deltas;
    }
    // #backpropagation(activations, zs, deltas) {
        
    //     console.log('Backprop');
    //     for (let layer_index = this.num_layers - 2; layer_index >= 0; layer_index--) {

    //         const currentLayer = this.layers[layer_index];

    //         const next_delta = deltas[layer_index + 1];
    //         const next_weights = this.weights[layer_index + 1];

    //         console.log(
    //             "Backprop at layer:",
    //             currentLayer.layer_name,
    //             "next_delta length:",
    //             next_delta.length
    //         );

    //         const { current_delta } =
    //             currentLayer.backpropagate(
    //                 this.onGPU,
    //                 next_weights,
    //                 next_delta,
    //                 zs,
    //                 layer_index,
    //                 currentLayer,
    //                 activations
    //             );

    //         deltas[layer_index] = current_delta;
    //     }

    //     return deltas;
    // }


    // forward propagation
    #Feedforward(input) {
        let current_input = input
        let all_layer_outputs = [input];
        let zs = [];

        let weights_biases_indexer = 0;
        for (let layer_index = 0; layer_index < this.num_layers; layer_index++) {
            const current_layer = this.layers[layer_index];
            const layer_weights = this.weights[weights_biases_indexer];
            const layer_biases = this.biases[weights_biases_indexer];

            const { outputs, z_values, incrementor_value } = current_layer.feedforward(this.onGPU, current_input, layer_weights, layer_biases, current_layer);
            weights_biases_indexer++;

            zs.push(z_values);
            current_input = outputs;
            all_layer_outputs.push(current_input);
        }

        return {
            predictions: current_input, 
            activations : all_layer_outputs,
            zs: zs
        };
    }
    //saving model
    #save(data, fileName, meta) {
        if (this.isfailed) {
            console.log('[FAILED]------- Failed to save model');
        }
        else {
            const dir = path.dirname(require.main.filename);

            const metadata = {
                "Date Created": `${new Date().toISOString().replace(/[:.]/g, '-')}`,
                "Number of epoch to train": meta[0],
                "Optimizer": meta[1],
                "Loss function": meta[2],
                "Task": meta[3],
                "Trained using": "Neurex",
                "Note": "This model can only be used on Neurex library. This cannot be used directly in other ML frameworks. DO NOT modify any of the parameters."
            };

            // Serialize and compress the model data
            const jsonString = JSON.stringify(data);
            const compressedData = zlib.deflateSync(jsonString);

            // Define file format:
            // [HEADER (4 bytes)] + [VERSION (1 byte)] + [DATA (compressed)]
            const header = Buffer.from("NRX2"); // Magic bytes
            const version = Buffer.from([0x02]); // Version 2

            // Combine all parts
            const finalBuffer = Buffer.concat([header, version, compressedData]);

            const nrxFilePath = path.join(dir, `${fileName}.nrx`);
            const metadataFilePath = path.join(dir, `metadata.json`);

            fs.writeFileSync(nrxFilePath, finalBuffer);
            fs.writeFileSync(metadataFilePath, JSON.stringify(metadata, null, 2));

            console.log(`[SUCCESS]------- Model is saved as ${fileName}.nrx`);
        }
    }

    #calculateClassificationAccuracy(predictions, actuals, taskType) {
        let correctPredictions = 0;
        for (let i = 0; i < predictions.length; i++) {
            let predictedLabel;
            let actualLabel;

            if (taskType === 'binary_classification') {
                predictedLabel = predictions[i][0] >= 0.5 ? 1 : 0;
                actualLabel = actuals[i][0]; // Assuming actuals are also arrays like [[0], [1]]
            } else if (taskType === 'multi_class_classification') {
                // Find the index of the maximum value in predictions for the predicted class
                predictedLabel = predictions[i].indexOf(Math.max(...predictions[i]));
                
                // If actuals[i] is an array with a single element (e.g., [0], [1]), it's integer-encoded.
                if (Array.isArray(actuals[i]) && actuals[i].length === 1) {
                    actualLabel = actuals[i][0]; // Directly take the integer label
                } else if (Array.isArray(actuals[i]) && actuals[i].length > 1) {
                    // Otherwise, assume one-hot encoded if it's an array with multiple elements (e.g., [1,0,0])
                    actualLabel = actuals[i].indexOf(1); 
                } else {
                    // Fallback for direct integer label if actuals[i] is not an array (e.g., 0, 1, 2 directly)
                    // This case might not be hit if Y_train is always provided as arrays of arrays.
                    actualLabel = actuals[i]; 
                }
            }

            if (predictedLabel === actualLabel) {
                correctPredictions++;
            }
        }
        return (correctPredictions / predictions.length) * 100;
    }

    #ifOneHotEndcoded(Y_train) {
        /**
        Checks if all rows in Y_train are one-hot encoded.
        Each row must:
        - Contain only 0s and 1s
        - Have exactly one "1"
        */
        for (let i = 0; i < Y_train.length; i++) {
            const row = Y_train[i];
            if (!Array.isArray(row)) return false;

            let onesCount = 0;
            for (let j = 0; j < row.length; j++) {
                if (row[j] !== 0 && row[j] !== 1) return false;
                if (row[j] === 1) onesCount++;
            }

            if (onesCount !== 1) return false;
        }
        return true;
    }
}

module.exports = Neurex;
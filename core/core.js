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
const optimizers = require('../optimizers')
const lossFunctions = require('../loss_functions');
const color = require('../prettify');
const { calculateTensorShape, XavierInitialization } = require('../utils');
const Layers = require('../layers/layers');



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
        this.input_shape = null;
        this.output_shape = [];
        this.currentShape = null;
        this.currentSize = null;
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

        // Optimizer state for each layer (weights and biases)
        this.optimizerStates = {
            weights: [],
            biases: []
        };

        this.onGPU = false;
        this.isfailed = false;
        this.weightGrads = [];
        this.biasGrads = [];

        this.checkpoint = 0; // if set to N, then every N of epochs will save the model, even if it's not yet fully train. Default is 0

    }

    /**
    * @typedef {Object} NeurexConfig
    * @property {number} [learning_rate] - Learning rate for training.
    * @property {string} [optimizer] - Optimizer to use [available: sgd, adam, adagrad, rmsprop, adadelta ].
    * @property {number} [randMin] - Minimum value for random initialization of weights/biases.
    * @property {number} [randMax] - Maximum value for random initialization of weights/biases.
    * @property {number} [checkpoint_per_epoch] - set a checkpoint per N epochs. Once set, every N epochs will save the model, even not yet fully trained.
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

        if (configs.checkpoint_per_epoch < 0) {
            this.isfailed = true;
            throw new Error(`${color.red}[Error]------- checkpoint cannot be less than 0. ${color.reset}`)
        }

        if (configs.checkpoint_per_epoch !== undefined) this.checkpoint = configs.checkpoint_per_epoch; 
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
            const layerType = layer.layer_name || newLayer.layer_name;
            const activationName = layer.activation_function ? layer.activation_function.name : 'None';

            if (layerType === 'convolutionalLayer') {
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
        const total_weights = this.weights.reduce((sum, arr) => sum + arr.length, 0);
        const total_biases = this.biases.reduce((sum, arr) => sum + arr.length, 0);
        console.log("===============================================================");
        console.log("Total layers: " + this.num_layers);
        console.log("Total Learnable parameters:",parseInt(total_weights+total_biases));
        console.log("===============================================================");
        
    }

    /**
     * Get the input shape
     *
     * @returns tensor input shape
     */
    getTensorShape() {
        return this.input_shape;
    }

    /**
     * Get the input size
     * 
     * @returns the input size equivalent of number of features as innput
     */
    getInputSize() {
        return this.input_size;
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
                backpropagate: layer.backpropagate,
                padding: layer.padding || '',
                filters: layer.filters || 0,
                strides: layer.strides || 0,
                kernel_size: layer.kernel_size || [0, 0],
                weightShape: layer.weightShape || [],
                inputShape: layer.inputShape || [],
                outputShape: layer.outputShape || [],

            })),
            "input_size":this.input_size,
            "input_shape":this.input_shape,
            "output_size":this.output_size,
            "num_layers":this.num_layers,
            "weights":this.weights.map(w => Array.from(w)),
            "biases":this.biases.map(b => Array.from(b)),
            "weightGrads":this.weightGrads.map(wg => Array.from(wg)),
            "biasGrads":this.biasGrads.map(bg => Array.from(bg)),
            
        };

        this.#save(data, fileName);
        
    }

    /**
     * 
     * @param {String} model - filename of the save model 
     */
    loadSavedModel(model) {
        try {
            if (!model) {
                throw new Error(`${color.red}\n[ERROR]------- No model provided ${color.red}`);
            }

            if (this.layers.length > 0) {
                this.isfailed = true;
                throw new Error(`${color.red}[ERROR]------- Failed to load model.\nReason:\nThere's already a new network being built. ${color.reset}`);
            }

            const dir = process.cwd();
            const model_file = path.join(dir, `${model}`);

            // Check extension
            if (path.extname(model_file) !== '.nrx') {
                throw new Error(`${color.red}Invalid file type. Only .nrx model files are supported.${color.reset}`);
            }

            // Read file
            const rawBuffer = fs.readFileSync(model_file);

            // Validate magic header
            const header = rawBuffer.slice(0, 4).toString('utf-8');
            if (header !== 'NRX3') {
                throw new Error(`${color.red}Invalid version format.${color.reset}`);
            }

            // Check version
            const version = rawBuffer[4];
            if (version !== 0x03) {
                throw new Error(`${color.red}Unsupported NRX version: ${version}${color.reset}`);
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
            this.input_size = modelData.input_size;
            this.output_size = modelData.output_size;
            this.num_layers = modelData.num_layers;
            this.weights = modelData.weights.map(w => new Float32Array(w));
            this.biases = modelData.biases.map(b => new Float32Array(b));
            this.optimizer = modelData.optimizer;
            this.weightGrads = modelData.weightGrads.map(wg => new Float32Array(wg));
            this.biasGrads = modelData.biasGrads.map(bg => new Float32Array(bg));
            this.input_shape = modelData.input_shape
            const layerBuilder = new Layers();
            this.layers = modelData.layers.map(layerData => {
                let newLayer;
                if (layerData.layer_name === "connected_layer") {
                    // Recreate the connected layer with the correct activation and size
                    newLayer = layerBuilder.connectedLayer(layerData.activation_function_name, layerData.layer_size);
                    newLayer.weightShape = layerData.weightShape;
                } else if (layerData.layer_name === "input_layer") {
                    // Recreate the input layer. Note: The input layer doesn't have methods, so this is just for consistency
                    newLayer = layerBuilder.inputShape({ features: layerData.layer_size });
                } else if (layerData.layer_name === "convolutionalLayer") {
                    // recreate Convolutional layer
                    newLayer = layerBuilder.convolutionalLayer(layerData.filters, layerData.strides, layerData.kernel_size, layerData.activation_function_name, layerData.padding);
                    newLayer.weightShape = layerData.weightShape;
                    newLayer.inputShape = layerData.inputShape;
                    newLayer.outputShape = layerData.outputShape;
                } else {
                    throw new Error(`${color.red}[ERROR] Unknown layer type '${layerData.layer_name}' found in model.${color.reset}`);
                }
                
                return newLayer;
            });
            
            this.#recalculateShape();
            console.log(`${color.lime}\n[SUCCESS]------- Model ${model} successfully loaded\n${color.reset}`);
        } catch (error) {
            console.log(error.message);
            process.exit(1);
        }
    }

    get_task_type() {
        return this.task || "Task not specified";
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

            if (this.layers.length > 0) {
                console.log(`\n${color.orange}[INFO]------- Skipping sequential build: \n\n reason:\n There/you might have loaded a model already. Please check if already load a model.\n${color.reset}`);
                return;
            }

            if (!layer_data || layer_data.length < 2) {
                throw new Error(`${color.red}[ERROR]------- No layers${color.reset} added.`);
            }

            layer_data.forEach(layer => {
                // extract input size
                if (layer.layer_name === "input_layer") {
                    this.input_size = layer.layer_size;
                    this.input_shape = layer.input_shape || [1, 1, this.input_size || 0];
                    this.depth = this.input_shape[2] || 0;

                    this.currentShape = [this.input_shape[0],this.input_shape[1], this.input_shape[2]];
                    this.currentSize = this.input_shape[0] * this.input_shape[1] * this.input_shape[2];

                }
                else {
                    this.layers.push(layer);
                }
            });

            this.hasSequentiallyBuild = true;
            this.num_layers = this.layers.length;
            this.#recalculateShape();
            this.#build();
            
            return layer_data; 
        }

        
        catch(err) {
            console.error(err);
        }
    }

    /**
     * @method pop - Removes the last layer of the model including it's initialzed or trained parameters and optimizer states. Useful for transfer learning
     * @throws {Error} - if there are no layers
     */
    pop() {
        if (this.layers.length == 0) throw new Error(`${color.red}[ERROR]-------- No layers has been added${color.reset}`);

        // get the last index
        const index = this.layers.length - 1;

        this.layers.splice(index, 1);
        this.weights.splice(index, 1);
        this.weightGrads.splice(index, 1);
        this.biases.splice(index, 1);
        this.biasGrads.splice(index, 1);

        this.num_layers--;
        this.#recalculateShape();
    }


    /**
     * @method add_layer
     * @param {Object} layer_data - layer data returned from Layers class
     *
     * @example
     * // sample usage
     * nrx.add_layer(layer.connectedLayer("relu", 10));
     */
    add_layer(layer_data) {

        if (this.layers.length == 0) throw new Error(`${color.red}[ERROR]-------- No layers has been added${color.reset}`);

        this.num_layers++;

        this.layers.push(layer_data);

        this.#buildSingle(layer_data);

    }

    /**
    * Trains the neural network using the provided training data, target values, number of epochs, and learning rate.
    * This method initializes the weights and biases for each layer, then iteratively performs forward propagation,
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
        *    layer.inputShape({features: 2}),
        *    layer.connectedLayer("relu", 3),
        *    layer.connectedLayer("relu", 3),
        *    layer.connectedLayer("softmax", 2)
        * ]);
        *
        *
        * model.train(X_train, Y_train, 'categorical_cross_entropy', 2000, 12);
    * * After training, you can use the network for predictions
    */

    async train(inputs, trainY, loss, epoch, batch_size = 1) {
        
        
        if (this.layers.length == 0) throw new Error(`${color.red}[ERROR]------- No layers constructed ${color.reset}`);

        let trainX = [];

        for (let i = 0; i < inputs.length; i++) {
            // the inputs must be in float32array, if any of the inputs are not in float32array, they'll be converted to float32array, otherwise, pass the input

            if (inputs[i].length != (this.input_shape[0] * this.input_shape[1] * this.input_shape[2]) || inputs[i].length != this.input_size) {
                this.isfailed = true;
                console.log(`${color.red}[ERROR]------- Input data must be the same shape set in the input layer${color.reset}\n- Use getTensorShape() or getInputSize()\n\nInput size/shape: ${inputs[i].length} || Expected: [${this.input_shape}] or ${this.input_size}\n`)
                throw new Error(`${color.red}Shape mismatch${color.reset}`);
            }

            trainX.push(inputs[i] instanceof Float32Array ? inputs[i] : new Float32Array(inputs[i].flat(Infinity)));
        }

        const lastLayerObject = this.layers[this.layers.length - 1];
        this.output_size = lastLayerObject.layer_size;

        try {
            if (!trainX || trainX.length == 0 || !trainY || trainY.length == 0 || !loss) {
                this.isfailed = true;
                console.error(`\n${color.red}Error${color.reset}`);
                console.log(`Train X: ${trainX ? "has data" : "no data"}`);
                console.log(`Train Y: ${trainY ? "has data" : "no data"}`);
                console.log(`Loss: ${loss ? "specified" : "not specified"}`);
                console.log(`Epoch: ${epoch ? "specified" : "not specified"}`);
                console.log(`Batch Size: ${batch_size ? "specified" : "not specified"}`);
                throw new Error(`[FAILED]------- There is/are missing parameter/s. Failed to start training...`);
            }

            if (epoch == 0 || batch_size == 0 || !epoch || !batch_size || epoch < 0 || batch_size < 0) {
                this.isfailed = true;
                throw new Error("[FAILED]------- Epoch or batch size cannot be zero or a negative number");
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
                        throw new Error(`${color.red}[ERROR]------- Output shape mismatch. The size of the output layer must be the same number of classes${color.reset}`);
                    }
                });

                // check if the Y_train is not one-hot encoded
                const isOneHotEncoded = this.#ifOneHotEndcoded(trainY);
                if (isOneHotEncoded) {
                    this.task = 'multi_class_classification';
                }
                else {
                    this.isfailed = true;
                    throw new Error(`${color.red}[ERROR]------- Y_train must be one-hot encoded for the categorical_cross_entropy loss. Use 'sparse_categorical_cross_entropy' instead if the Y_train is interger-encoded labels.${color.reset}`);
                }
                
            }
            else if (lastLayerActivation === 'softmax' && lossLower === 'sparse_categorical_cross_entropy') {
                this.task = 'multi_class_classification';
            }
            else {
                this.isfailed = true;
                throw new Error(`${color.red}[ERROR]------- Using ${lossLower} having output size of ${this.output_size} and a ${lastLayerActivation} function in the output layer is currently unavailable for this core's task.${color.reset}`);
            }

            if (!optimizerFn) {
                this.isfailed = true;
                throw new Error(`${color.red}Unknown optimizer: ${this.optimizer} ${color.reset}`)
            };

            


            console.log(`${color.orange}\n[TASK]------- Training session is starting${color.reset}\n`);

            const totalBatches = Math.ceil(trainX.length / batchSize);
            let logMessage;
            // epoch loop
            for (let current_epoch = 0; current_epoch < epoch; current_epoch++) {
                let totalepochLoss = 0;
                let numBatches = 0; // Added to count batches

                // batch size
                for (let batchStart = 0; batchStart < trainX.length; batchStart += batchSize) {
                    numBatches++; // Increment batch count
                    const currentBatch = Math.floor(batchStart / batchSize) + 1;

                    const batchEnd = Math.min(batchStart + batchSize, trainX.length);
                    const actualBatchSize = batchEnd - batchStart;

                    this.#reinitiateWeightSBiasGrads(); // reset to grads (weights and biases grads) to 0s

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
                                    throw new Error(`${color.red}[ERROR]------- Uknown loss function for multi-class classification loss. Loss: ${lossLower} is unknown.${color.reset}`)
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

                        deltas[output_layer_index] = new Float32Array(dOutputlayer);


                        // backpropagation loop
                        const allDeltas = this.#backpropagation(activations, zs, deltas);


                        // === STEP 3: Accumulate Gradients === //
                        for (let l = 0; l < this.num_layers; l++) {
                            const delta = allDeltas[l];
                            const a_prev = activations[l];
                            const layer_data_obj = this.layers[l];


                            // Accumulate weight gradients
                            weightGrads[l] = layer_data_obj.computeWeightGradients(a_prev, delta, weightGrads[l], layer_data_obj);


                            // Accumulate bias gradients
                            biasGrads[l] = layer_data_obj.computeBiasGradients(biasGrads[l], delta, layer_data_obj);

                        }
                    }

                    batchLoss /= actualBatchSize;
                    totalepochLoss += batchLoss;
                    logMessage = `[Epoch] ${current_epoch + 1}/${epoch} ` +`| [Batch] ${currentBatch}/${totalBatches} ` +`| [Batch Loss]: ${batchLoss.toFixed(6)} `
                    process.stdout.write(`\r`+logMessage);


                    // Divide accumulated gradients by the actual batch size and use the optimizer function to update the paramters
                    for (let l = 0; l < this.num_layers; l++) {
                        
                        const layer_data_obj = this.layers[l];

                        // scale weight gradients
                        weightGrads[l] = layer_data_obj.scaleGrads(weightGrads[l], actualBatchSize, layer_data_obj);

                        // scale bias gradients
                        biasGrads[l] = layer_data_obj.scaleGrads(biasGrads[l], actualBatchSize);

                        // update Weights
                        const res1 = optimizerFn(this.weights[l], weightGrads[l], this.optimizerStates.weights[l], this.learning_rate);

                        // Update biases
                        const res2 = optimizerFn(this.biases[l], biasGrads[l], this.optimizerStates.biases[l], this.learning_rate);

                        // assigned updated weights to it's current index position relative to the layer's index
                        this.weights[l] = res1.params;

                        // assigned updated biases to it's current index position relative to the layer's index
                        this.biases[l] = res2.params;

                        // assigned updated weight states to it's current index position relative to the layer's index
                        this.optimizerStates.weights[l] = res1.state;

                        // assigned updated bias states to it's current index position relative to the layer's index
                        this.optimizerStates.biases[l] = res2.state;
                    }

                }

                let AverageEpochLoss = totalepochLoss / numBatches; 
                let setColor = AverageEpochLoss > 0.9 ? color.red : 
                                AverageEpochLoss > 0.5 ? color.orange :
                                AverageEpochLoss > 0.1 ? color.yellow :
                                AverageEpochLoss > 0.03 ? color.lime : color.green;
                
                logMessage += `| [Epoch Loss]: ${setColor} ${AverageEpochLoss.toFixed(7)} ${color.reset}`;

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
                process.stdout.write('\r'+logMessage);
                // if the checkpoint is not 0 (assume it was configured), proceed to saving the model after showing the latest training information
                if (this.checkpoint > 0 && (current_epoch + 1) % this.checkpoint === 0) {
                    console.log();
                    console.log(`\n${color.orange}🗼 [CHECKPOINT] Saving at epoch ${current_epoch + 1}... 🗼${color.reset}`);
                    this.saveModel(`Checkpoint_Epoch_${current_epoch + 1}`);
                }
                console.log();
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

            for (let i = 0; i < input.length; i++) {
                if (input[i].length != (this.input_shape[0] * this.input_shape[1] * this.input_shape[2]) || input[i].length != this.input_size) {
                    this.isfailed = true;
                    console.log(`${color.red}[ERROR]------- Input data must be the same shape set in the input layer${color.reset}\n- Use getTensorShape() or getInputSize()\n\nInput size/shape: ${input[i].length} || Expected: [${this.input_shape}] or ${this.input_size}\n`)
                    throw new Error(`${color.red}Shape mismatch${color.reset}`);
                }
            }            

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
            // Start from the base input state
            let [H, W, D] = this.input_shape;
            this.currentShape = [H, W, D];
            this.currentSize = H * W * D;

            this.layers.forEach((layer_data) => {
                if (layer_data.layer_name === "connected_layer") {
                    const inputSize = this.currentSize;
                    const outputSize = layer_data.layer_size;
                    const TotalWeightSize = outputSize * inputSize;

                    const weights = new Float32Array(TotalWeightSize);
                    const weightGrads = new Float32Array(TotalWeightSize);
                    const biases = new Float32Array(outputSize);
                    const biasGrads = new Float32Array(outputSize);

                    const limit = XavierInitialization(inputSize, outputSize);


                    for (let i = 0; i < TotalWeightSize; i++) {
                        weights[i] = (Math.random() * 2 - 1) * limit;
                    }

                    for (let i = 0; i < outputSize; i++) {
                        biases[i] = (Math.random() * 2 - 1) * limit;
                    }

                    this.weights.push(weights);
                    this.biases.push(biases);
                    this.weightGrads.push(weightGrads);
                    this.biasGrads.push(biasGrads);

                    this.currentShape = [1, 1, outputSize];
                    this.currentSize = outputSize;
                    
                    layer_data.weightShape = [inputSize, outputSize];
                } 
                
                else if (layer_data.layer_name === "convolutionalLayer") {

                    const filters = layer_data.filters;
                    const [kH, kW] = layer_data.kernel_size;
                    const stride = layer_data.strides || 1;
                    const padding = layer_data.padding || "same";

                    const inputH = this.currentShape[0];
                    const inputW = this.currentShape[1];
                    const inputDepth = this.currentShape[2];

                    layer_data.inputShape = [inputH, inputW, inputDepth];

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

                    this.weights.push(kernels);
                    this.weightGrads.push(kernelGrads);
                    this.biases.push(biases);
                    this.biasGrads.push(biasGrads);

                    // Calculate output shape
                    const { OutputHeight, OutputWidth, CalculatedTensorShape } = calculateTensorShape(inputH, inputW, kH, kW, filters, stride, padding);

                    // store output shape too
                    layer_data.outputShape = [OutputHeight, OutputWidth, filters];

                    this.currentShape = [OutputHeight, OutputWidth, filters];
                    this.currentSize = CalculatedTensorShape;

                    layer_data.weightShape = [filters, kH, kW, inputDepth];
                }
            });

            this.hasBuilt = true;
        } catch (error) {
            console.error(`${color.red}[BUILD ERROR]------- ${error.message}${color.reset}`);
            throw error;
        }
    }

    #buildSingle(layer_data) {
        const layer_index = this.layers.length - 1;

        if (layer_data.layer_name === "connected_layer") {
            const inputSize = this.currentSize;
            const outputSize = layer_data.layer_size;
            const totalWeightSize = inputSize * outputSize;

            // Initialize flat Float32Arrays
            const weights = new Float32Array(totalWeightSize);
            const weightGrads = new Float32Array(totalWeightSize);
            const biases = new Float32Array(outputSize);
            const biasGrads = new Float32Array(outputSize);

            const limit = XavierInitialization(inputSize, outputSize);
            for (let i = 0; i < totalWeightSize; i++) {
                weights[i] = (Math.random() * 2 - 1) * limit;
            }

            this.weights.push(weights);
            this.biases.push(biases);
            this.weightGrads.push(weightGrads);
            this.biasGrads.push(biasGrads);

            layer_data.weightShape = [inputSize, outputSize];
            this.currentShape = [1, 1, outputSize];
            this.currentSize = outputSize;
        } 
        
        else if (layer_data.layer_name === "convolutionalLayer") {
            const [H, W, D] = this.currentShape;
            const filters = layer_data.filters;
            const [kH, kW] = layer_data.kernel_size;
            const stride = layer_data.strides || 1;
            const padding = layer_data.padding || "same";

            const totalSize = filters * kH * kW * D;
            const kernels = new Float32Array(totalSize);
            const kernelGrads = new Float32Array(totalSize);
            const biases = new Float32Array(filters);
            const biasGrads = new Float32Array(filters);

            // Initialization
            const limit = XavierInitialization(kH * kW * D, kH * kW * filters);
            for (let i = 0; i < totalSize; i++) {
                kernels[i] = (Math.random() * 2 - 1) * limit;
            }

            this.weights.push(kernels);
            this.biases.push(biases);
            this.weightGrads.push(kernelGrads);
            this.biasGrads.push(biasGrads);

            // Calculate output shape
            const { OutputHeight, OutputWidth, CalculatedTensorShape } = calculateTensorShape(H, W, kH, kW, filters, stride, padding);
            
            layer_data.inputShape = [H, W, D];
            layer_data.outputShape = [OutputHeight, OutputWidth, filters];
            layer_data.weightShape = [filters, kH, kW, D];

            this.currentShape = [OutputHeight, OutputWidth, filters];
            this.currentSize = CalculatedTensorShape;
        }
    }

    // backpropagation
    #backpropagation(activations, zs, deltas_array) {
        // The loop should iterate from the second-to-last layer (this.num_layers - 2) down to the first layer (0).

        let deltas = deltas_array;
        let delta_indexer = this.num_layers - 2;
        for (let layer_index = delta_indexer; layer_index >= 0; layer_index--) {
            
            const next_weights = this.weights[layer_index + 1];
            const nextLayer = this.layers[layer_index + 1];
            const next_delta = deltas[layer_index + 1];
            
            // Get the current layer object, which now holds its backpropagation logic
            const currentLayer = this.layers[layer_index];

            // Call the layer's specific backpropagation method
            const {current_delta, decrementor_value} = currentLayer.backpropagate(this.onGPU,next_weights, next_delta, zs, layer_index, currentLayer, this.weights, activations, nextLayer, this.layers);
            delta_indexer--;
            
            deltas[layer_index] = current_delta;
        }

        return deltas;
    }


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
    #save(data, fileName) {
        if (this.isfailed) {
            console.log('[FAILED]------- Failed to save model');
        }
        else {
            const dir = process.cwd() //path.dirname(require.main.filename);

            // Serialize and compress the model data
            const jsonString = JSON.stringify(data);
            const compressedData = zlib.deflateSync(jsonString);

            // Define file format:
            // [HEADER (4 bytes)] + [VERSION (1 byte)] + [DATA (compressed)]
            const header = Buffer.from("NRX3"); // Magic bytes
            const version = Buffer.from([0x03]); // Version 3

            // Combine all parts
            const finalBuffer = Buffer.concat([header, version, compressedData]);

            const nrxFilePath = path.join(dir, `${fileName}.nrx`);

            fs.writeFileSync(nrxFilePath, finalBuffer);

            console.log(`[SUCCESS]------- Model is saved as ${fileName}.nrx\n`);
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

    #recalculateShape() {
        let H = this.input_shape[0];
        let W = this.input_shape[1];
        let D = this.input_shape[2];

        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];

            if (layer.layer_name === "convolutionalLayer") {
                const filters = layer.filters;
                const [kernelHeight, kernelWidth] = layer.kernel_size;
                const stride = layer.strides || 1;
                const padding = layer.padding || "same";

                const {OutputHeight, OutputWidth} = calculateTensorShape(H, W, kernelHeight, kernelWidth, D, stride, padding);
                H = OutputHeight;
                W = OutputWidth;
                D = filters;
                
            }
            else if (layer.layer_name === "connected_layer") {
                H = 1;
                W = 1;
                D = layer.layer_size;
            }
        }

        this.currentShape = [H, W, D];
        this.currentSize = H * W * D;
    }

    #reinitiateWeightSBiasGrads() {
        for (let l = 0; l < this.num_layers; l++) {
            if (this.weightGrads[l]) {
                this.weightGrads[l].fill(0);
            }
            if (this.biasGrads[l]) {
                this.biasGrads[l].fill(0);
            }

        }
    }
}

module.exports = Neurex;
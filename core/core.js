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
const color = require('../color-code');
const { calculateTensorShape, XavierInitialization, getTotalMB, formatDuration, calculateTransConvOutputShape } = require('../utils');
const Layers = require('../layers/layers');
const { onFloat32Module, modeConfiguration } = require('../gpu/modeSelector');
const { init, gradientClipping, scaleGrads } = require('./bindings');
const { setGlobalParams } = require('../gpu/globals');
const version = require('../package.json').version;

class Neurex {
    constructor () {
        this.weights = [];
        this.biases = [];
        this.output_layers_templates = [];
        this.num_layers = 0;
        this.input_size = 1;
        this.input_shape = [1, 1, 1];
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
        this.optimizer = null;
        this.learning_rate = 0.001;
        this.initial_learning_rate = 0.001;
        this.lr_scheduler = null;
        this.clip_norm_value = 1.0;
        this.onChange_optimizer = null;
        this.plugins = null;

        // Optimizer state for each layer (weights and biases)
        this.optimizerStates = {
            weights: [],
            biases: []
        };

        this.isfailed = false;
        this.weightGrads = [];
        this.biasGrads = [];

        this.checkpoint = 0; // if set to N, then every N of epochs will save the model, even if it's not yet fully train. Default is 0
        this.isInit = false;

        this.parametric_layers = [];
        this.miscellaneous = null;
        
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
        
        if (configs.learning_rate !== undefined) {
            this.learning_rate = configs.learning_rate;
            this.initial_learning_rate = configs.learning_rate;
        }
        if (configs.lr_scheduler !== undefined) this.lr_scheduler = configs.lr_scheduler || null;

        if (configs.checkpoint_per_epoch < 0) {
            this.isfailed = true;
            throw new Error(`${color.red}[Error]------- checkpoint cannot be less than 0. ${color.reset}`)
        }

        if (configs.checkpoint_per_epoch !== undefined) this.checkpoint = configs.checkpoint_per_epoch;

        if (configs.clip_norm_value !== undefined) this.clip_norm_value = configs.clip_norm_value || 1.0;

        // mode: gpu | cpu | auto
        // onFLoat32Module: true | false

        modeConfiguration(configs.mode || "cpu");
        onFloat32Module(configs.onFLoat32Module || false);

        this.optimizer = configs.optimizer || optimizers.SGD();
        
        if (configs.onChange_optimizer !== undefined) {
            this.onChange_optimizer = {
                targetEpoch: configs.onChange_optimizer.targetEpoch,
                optimizer: configs.onChange_optimizer.optimizer
            }
        }

        if (configs.plugins !== undefined) {
            this.plugins = configs.plugins;
        }

        init();
        this.isInit = true;
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
        if (!this.layers || this.layers.length === 0) {
            console.error(`${color.red}[ERROR]------- An error occurred${color.reset}`);
            throw new Error('No layers to show details');
        }

        const COLS = [
            { title: 'Layer (type)', width: 24 },
            { title: 'Output Shape', width: 26 },
            { title: 'Activation',   width: 14 },
            { title: 'Parameters',   width: 20 },
            { title: 'Padding',      width: 12 },
        ];

        const totalWidth = COLS.reduce((s, c) => s + c.width, 0);

        const row = (cells) => cells.map((c, i) => String(c).padEnd(COLS[i].width)).join('');
        const hr = (ch) => ch.repeat(totalWidth);
        const center = (text) => {
            const pad = Math.max(0, Math.floor((totalWidth - text.length) / 2));
            return ' '.repeat(pad) + text;
        };

        console.log(hr('_'));
        console.log(center('Model Summary'));
        console.log(hr('_'));
        console.log(`Input size: ${this.input_size}`);
        console.log(`Input Shape: [${this.input_shape}]`);
        console.log(`Number of layers: ${this.num_layers}`);
        console.log(hr('-'));
        console.log(row(COLS.map(c => c.title)));
        console.log(hr('='));

        let pointer = 0;
        this.layers.forEach((layer) => {
            const layerType = layer.layer_name;
            const activationName = layer.activation_function ? layer.activation_function.name : 'None';

            const isParametric = this.parametric_layers.includes(layerType);

            let paramCount = 0;
            if (isParametric) {
                const w = this.weights[pointer] ? this.weights[pointer].length : 0;
                const b = this.biases[pointer] && !this.biases[pointer].every(v => v == 0)  ? this.biases[pointer].length : 0;
                paramCount = w + b;
                pointer++;
            }

            let displayName, outputShape, activation, params, padding;

            switch (layerType) {
                case 'convolutionalLayer':
                    displayName = 'Convolutional Layer';
                    outputShape = `(${layer.outputShape.join(' x ')})`;
                    activation = activationName;
                    params = paramCount.toLocaleString();
                    padding = layer.padding || 'None';
                    break;

                case 'transConvLayer':
                    displayName = 'Trans Convolution';
                    outputShape = `(${layer.outputShape.join(' x ')})`;
                    activation = activationName;
                    params = paramCount.toLocaleString();
                    padding = layer.padding || 'None';
                    break;

                case 'connected_layer':
                    displayName = 'Connected Layer';
                    outputShape = `(1 x 1 x ${layer.layer_size})`;
                    activation  = activationName;
                    params  = paramCount.toLocaleString();
                    padding = layer.padding || 'None';
                    break;

                case 'maxPooling':
                    displayName = 'Max Pooling';
                    outputShape = `(${layer.outputShape.join(' x ')})`;
                    activation  = 'None';
                    params = '0 (non-param)';
                    padding = layer.padding || 'None';
                    break;
                case "EmbeddingLayer":
                    displayName = "Embedding Layer";
                    outputShape = `(${layer.outputShape.join(' x ')})`;
                    activation = 'None';
                    params = paramCount.toLocaleString();
                    padding = "None"
                    break;
                case "recurrent_cell":
                    displayName = "Recurrent Cell";
                    outputShape = `(${layer.outputShape.join(' x ')})`;
                    activation = activationName|| 'None';
                    params = paramCount.toLocaleString();
                    padding = "None"
                    break;
                default:
                    displayName = layerType;
                    outputShape = '-';
                    activation  = activationName;
                    params = paramCount.toLocaleString();
                    padding = layer.padding || 'None';
            }

            console.log(row([displayName, outputShape, activation, params, padding]));
        });

        // 5) Footer block.
        const totalWeights = this.weights.reduce((sum, arr) => sum + arr.length, 0);
        const totalBiases  = this.biases.reduce((sum, arr) => sum + arr.length, 0);
        const totalSizeMB  = getTotalMB(this.weights) + getTotalMB(this.biases);

        console.log(hr('='));
        console.log(`Total learnable parameters: ${(totalWeights + totalBiases).toLocaleString()}`);
        console.log(`Total size (MegaBytes): ${totalSizeMB.toFixed(2)} MB`);
        console.log(hr('='));
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
     * @method get_miscellaneous_data
     * @returns {Object} Saved miscellaneous data upon model saving
     */
    get_miscellaneous_data() {
        return this.miscellaneous;
    }

    /**
     * 
     * saveModel() allows you to save your model's architecture, weights, and biases, as well as other parameters. The model will be exported
     *  as a .nrx (neurex) model
     * @method saveModel()
     * @param {string} modelName the filename of your model
     * @param {Object} miscellaneous data that can be included to be saved in the model. Note: This may increase the model size when adding miscellaneous.
     *   
     */
    saveModel(modelName = null, miscellaneous) {
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
            "learning_rate":this.learning_rate,
            "input_size":this.input_size,
            "input_shape":this.input_shape,
            "output_size":this.output_size,
            "num_layers":this.num_layers,
            "clip_norm_value": this.clip_norm_value,
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
                poolSize: layer.poolSize || [],
                embeddingDim: layer.embeddingDim || 1,
                vocabSize: layer.vocabSize || 1,
                maxSequenceLength: layer.maxSequenceLength || 1,
                units: layer.units || 1,
                return_sequence: layer.return_sequence || false,
                return_state: layer.return_state || false,
                isParametric: layer.isParametric
            })),
            "miscellaneous": miscellaneous
        };

        this.#save(data, this.weights, this.biases, fileName);
        
    }

    /**
     * 
     * @param {String} model path to your model
     * @param {Boolean} showLog outputs confirmation log when loading and successfullu loading a model. Default value is `true`. 
     */
    loadSavedModel(model, showLog = true) {
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

            if (showLog) {
                console.log(`${color.yellow}[INFO]------- Loading model from ${model_file}${color.reset}`)
            }
            

            // Check extension
            if (path.extname(model_file) !== '.nrx') {
                throw new Error(`${color.red}Invalid file type. Only .nrx model files are supported.${color.reset}`);
            }

            // Read file
            const rawBuffer = fs.readFileSync(model_file);

            // Validate magic header
            const header = rawBuffer.slice(0, 4).toString('utf-8');
            if (header !== 'NRX4') {
                throw new Error(`${color.red}Invalid version format.${color.reset}`);
            }

            // Check version
            const version = rawBuffer[4];
            if (version !== 0x04) {
                throw new Error(`${color.red}Unsupported NRX version: ${version}${color.reset}`);
            }

            // Read metadata block (small, JSON-friendly: arch, hyperparams, shapes)
            const metaLength = rawBuffer.readUInt32LE(5);
            const metaStart = 9;
            const metaCompressed = rawBuffer.slice(metaStart, metaStart + metaLength);
            const metaJson = zlib.inflateSync(metaCompressed).toString('utf-8');
            const modelData = JSON.parse(metaJson);

            // Read tensor block (weights/biases as raw concatenated Float32 bytes).
            // This is never run through JSON.parse/stringify; we slice typed array
            // views directly out of the decompressed buffer instead.
            const tensorCompressed = rawBuffer.slice(metaStart + metaLength);
            const tensorBlock = zlib.inflateSync(tensorCompressed);

            let byteOffset = 0;
            const readTensors = (lengths) => lengths.map(len => {
                const byteLen = len * Float32Array.BYTES_PER_ELEMENT;
                // tensorBlock's offset within its own underlying ArrayBuffer must be
                // included, since inflateSync may return a Buffer that is itself a
                // view into a larger pooled allocation.
                const arr = new Float32Array(
                    tensorBlock.buffer.slice(
                        tensorBlock.byteOffset + byteOffset,
                        tensorBlock.byteOffset + byteOffset + byteLen
                    )
                );
                byteOffset += byteLen;
                return arr;
            });

            const loadedWeights = readTensors(modelData.weightLengths);
            const loadedBiases = readTensors(modelData.biasLengths);

            // Assign properties
            this.miscellaneous = modelData.miscellaneous;
            this.initial_learning_rate = modelData.learning_rate || 0.001;
            this.task = modelData.task;
            this.loss_function = modelData.loss_function;
            this.epoch_count = modelData.epoch;
            this.batch_size = modelData.batch_size;
            this.learning_rate = modelData.learning_rate;
            this.input_size = modelData.input_size;
            this.output_size = modelData.output_size;
            this.num_layers = modelData.num_layers;
            this.weights = loadedWeights;
            this.biases = loadedBiases;
            this.input_shape = modelData.input_shape;
            this.clip_norm_value = modelData.clip_norm_value || 1.0;
            const layerBuilder = new Layers();
            this.layers = modelData.layers.map(layerData => {
                let newLayer;
                if (layerData.layer_name === "connected_layer") {
                    // Recreate the connected layer with the correct activation and size
                    newLayer = layerBuilder.connectedLayer(layerData.layer_size, layerData.activation_function_name);
                    newLayer.weightShape = layerData.weightShape;
                    this.output_layers_templates.push(new Float32Array(layerData.layer_size));
                    this.parametric_layers.push(layerData.layer_name);
                } else if (layerData.layer_name === "input_layer") {
                    // Recreate the input layer. Note: The input layer doesn't have methods, so this is just for consistency
                    newLayer = layerBuilder.inputShape({ features: layerData.layer_size });
                } 
                else if (layerData.layer_name === "convolutionalLayer") {
                    // recreate Convolutional layer
                    newLayer = layerBuilder.convolutionalLayer(layerData.filters, layerData.strides, layerData.kernel_size, layerData.activation_function_name, layerData.padding);
                    newLayer.weightShape = layerData.weightShape;
                    newLayer.inputShape = layerData.inputShape;
                    newLayer.outputShape = layerData.outputShape;
                    const [H, W, D] = layerData.outputShape;
                    const totalSize = H * W * D;
                    this.output_layers_templates.push(new Float32Array(totalSize));
                    this.parametric_layers.push(layerData.layer_name);
                } else if (layerData.layer_name === "maxPooling") {
                    newLayer = layerBuilder.maxPooling(layerData.poolSize, layerData.strides, layerData.padding);
                    newLayer.inputShape = layerData.inputShape;
                    newLayer.outputShape = layerData.outputShape;
                    const [H, W, D] = layerData.outputShape;
                    const totalSize = H * W * D;
                    this.output_layers_templates.push(new Float32Array(totalSize));
                }
                else if (layerData.layer_name === "EmbeddingLayer") {
                    const vocabSize = layerData.vocabSize;
                    const embeddingDim = layerData.embeddingDim;
                    const sequence_length = layerData.maxSequenceLength;
                    const outputSize = sequence_length * embeddingDim;
                    newLayer = layerBuilder.embeddingLayer(vocabSize, embeddingDim, sequence_length);
                    newLayer.inputShape = [];
                    newLayer.outputShape = [1, 1, outputSize];
                    newLayer.weightShape = [vocabSize, embeddingDim];
                    newLayer.outputSize = outputSize;
                    this.output_layers_templates.push(new Float32Array(outputSize));
                    this.parametric_layers.push(layerData.layer_name);
                }
                else if (layerData.layer_name === "recurrent_cell") {
                    const units = layerData.units;
                    const return_sequence = layerData.return_sequence || false;
                    const return_state = layerData.return_state || false;
                    newLayer = layerBuilder.recurrentCell(units, layerData.activation_function_name, return_sequence, return_state);
                    newLayer.weightShape = layerData.weightShape;
                    newLayer.inputShape = layerData.inputShape;
                    newLayer.outputShape = layerData.outputShape;
                    newLayer.maxSequenceLength = layerData.maxSequenceLength || 1;
                    this.output_layers_templates.push(new Float32Array(units));
                    this.parametric_layers.push(layerData.layer_name);
                }
                else {
                    throw new Error(`${color.red}[ERROR] Unknown layer type '${layerData.layer_name}' found in model.${color.reset}`);
                }
                
                return newLayer;
            });
            
            this.#recalculateShape();

            for (let i = 0; i < this.weights.length; i++) {
                this.weightGrads.push(new Float32Array(this.weights[i].length).fill(0));
                this.biasGrads.push(new Float32Array(this.biases[i].length).fill(0));
            }
            
            if (showLog) {
                console.log(`${color.lime}[SUCCESS]------- Model ${model} successfully loaded\n${color.reset}`);
            }
            
        } catch (error) {
            console.log(error);
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

            if (!layer_data || layer_data.length < 1) {
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
        if (this.layers.length === 0) throw new Error(`${color.red}[ERROR]-------- No layers has been added${color.reset}`);

        const index = this.layers.length - 1;
        const removedLayer = this.layers[index];

        this.layers.splice(index, 1);
        this.output_layers_templates.splice(index, 1);
        this.num_layers--;
        this.#recalculateShape();

        const parametric = this.parametric_layers;
        if (parametric.includes(removedLayer.layer_name)) {
            const weightIndex = this.layers.filter(l => parametric.includes(l.layer_name)).length;
            this.weights.splice(weightIndex, 1);
            this.weightGrads.splice(weightIndex, 1);
            this.biases.splice(weightIndex, 1);
            this.biasGrads.splice(weightIndex, 1);
        }
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
        if (!this.isInit) {
            init();
            this.isInit = true;
        }
    
        setGlobalParams(
            this.weights, 
            this.biases, 
            this.output_layers_templates,
        );

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

        // Infer task type based on output layer and loss/activation
        let lastLayer = this.layers[this.layers.length - 1];
        this.loss_function = loss.toLowerCase();
        const loss_function = lossFunctions[this.loss_function.toLowerCase()];
            
        this.epoch_count = epoch;
        this.batch_size = batch_size;
        const batchSize = batch_size;
            
        const lossLower = loss.toLowerCase();

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

            if (this.plugins?.trainingVisualizer) {
                const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
                for (const visualizer of this.plugins.trainingVisualizer) {
                    if (typeof visualizer?.initialize !== 'function') {
                        this.isfailed = true;
                        throw new Error("A plugin function doesn't have a callable initialize() function.");
                    }

                    await visualizer?.initialize();

                    await delay(5000);

                }
            }

            // Infer task type based on output layer and loss/activation
            let lastLayerObject = this.layers[this.layers.length - 1];
            // in order to support any layer to be an output layer, each layer type has their own way of determining inference type
            const taskType = lastLayerObject.determineInferenceType(lastLayerObject, lossLower, trainY);
            this.task = taskType;

            console.log(`${color.orange}\n[TASK]------- Training session is starting${color.reset}\n`);

            const totalBatches = Math.ceil(trainX.length / batchSize);
            let logMessage;
            let previousEpochLoss = 0;
            let startTime;
            // epoch loop
            for (let current_epoch = 0; current_epoch < epoch; current_epoch++) {
                
                if (this.lr_scheduler && current_epoch > 0) {
                    this.learning_rate = this.lr_scheduler({
                        current_epoch: current_epoch,
                        learning_rate: this.learning_rate,
                        previousEpochLoss: previousEpochLoss,
                        initial_learning_rate: this.initial_learning_rate,
                        batchSize: batchSize,
                        totalEpochs: epoch,
                        trainingFeatureSize: trainX[0].length
                    });
                }

                // this logic is for automated changing of optimizer mid training. 
                // changing optimizer must change before the target epoch so that the target epoch will use the optimizer
                if (this.onChange_optimizer && current_epoch > 0) {
                    if (current_epoch == this.onChange_optimizer.targetEpoch - 1) {
                        console.log(`\n${color.yellow}[Note]${color.reset} Changing optimizer on Epoch ${current_epoch+1}. Using ${color.yellow}${this.onChange_optimizer.optimizer.name || "Custom Optimizer"}${color.reset}...\n`)
                        this.optimizer = this.onChange_optimizer.optimizer;
                    }
                }

                startTime = performance.now();
                let totalepochLoss = 0;
                let numBatches = 0; // Added to count batches

                // batch size
                for (let batchStart = 0; batchStart < trainX.length; batchStart += batchSize) {
                    numBatches++; // Increment batch count
                    const currentBatch = Math.floor(batchStart / batchSize) + 1;

                    const batchEnd = Math.min(batchStart + batchSize, trainX.length);
                    const actualBatchSize = batchEnd - batchStart;

                    let weightGrads = this.weightGrads;

                    let biasGrads = this.biasGrads;

                    let batchLoss = 0;                  

                    // Accumulate gradients for each sample in the batch
                    for (let sample_index = batchStart; sample_index < batchEnd; sample_index++) {

                        let input = trainX[sample_index];
                        let actual = trainY[sample_index];

                        // feed forward
                        const {predictions, activations, zs} = this.#Feedforward(input);
                        let deltas = [];
                        let dOutputlayer = [];
                        

                        batchLoss += loss_function(predictions, actual);

                        // === STEP 1: Compute delta for output layer === //
                        let output_layer_index = this.num_layers - 1;
                        
                        deltas[output_layer_index] = lastLayerObject.getOutputLayerDelta(predictions, actual, zs, lossLower, this.task, lastLayerObject);

                        // === STEP 2: backpropagate the output layer delta === //
                        const {deltas:allDeltas} = this.#backpropagation(activations, zs, deltas);

                        // console.log(allDeltas);

                        // === STEP 3: Accumulate Gradients === //
                        let pointer = 0;
                        for (let l = 0; l < this.layers.length; l++) {
                            const delta = allDeltas[l];
                            const a_prev = activations[l];
                            const layer_data_obj = this.layers[l];

                            const parametric_layers = this.parametric_layers;

                            if (!parametric_layers.includes(layer_data_obj.layer_name)) {
                                continue;
                            }


                            // Accumulate weight gradients
                            weightGrads[pointer] = layer_data_obj.computeWeightGradients(a_prev, delta, weightGrads[pointer], layer_data_obj);

                            // Accumulate bias gradients
                            biasGrads[pointer] = layer_data_obj.computeBiasGradients(biasGrads[pointer], delta, layer_data_obj);
                            pointer++;
                        }
                    }

                    batchLoss /= actualBatchSize;
                    totalepochLoss += batchLoss;
                    logMessage = `[Epoch] ${current_epoch + 1}/${epoch} ` +`| [Batch] ${currentBatch}/${totalBatches} ` +`| [Batch Loss]: ${batchLoss.toFixed(6)} `
                    process.stdout.write(`\r`+logMessage);


                    let pointer = 0;
                    // Divide accumulated gradients by the actual batch size and use the optimizer function to update the paramters
                    for (let l = 0; l < this.num_layers; l++) {
                        
                        const layer_data_obj = this.layers[l];

                        const parametric_layers = this.parametric_layers;

                        if (!parametric_layers.includes(layer_data_obj.layer_name)) {
                            continue;
                        }

                        // scale weight gradients
                        weightGrads[pointer] = scaleGrads(weightGrads[pointer], actualBatchSize, layer_data_obj);

                        // scale bias gradients
                        biasGrads[pointer] = scaleGrads(biasGrads[pointer], actualBatchSize);
                        
                        // clip accumulated weight gradients using a threshold
                        weightGrads[pointer] = gradientClipping(weightGrads[pointer], this.clip_norm_value);

                        // clip accumulated bias gradients using a threshold
                        biasGrads[pointer] = gradientClipping(biasGrads[pointer], this.clip_norm_value);

                        // update Weights using the optimizer
                        const res1 = this.optimizer({params: this.weights[pointer], grads: weightGrads[pointer], state: this.optimizerStates.weights[pointer], lr: this.learning_rate, previousEpochLoss: previousEpochLoss, current_epoch: current_epoch, batchSize: batchSize, totalEpoch: epoch, trainingFeatureSize: trainX[0].length});

                        // Update biases using the optimizer
                        const res2 = this.optimizer({params: this.biases[pointer], grads: biasGrads[pointer], state: this.optimizerStates.biases[pointer], lr: this.learning_rate, previousEpochLoss: previousEpochLoss, current_epoch: current_epoch, batchSize: batchSize, totalEpoch: epoch, trainingFeatureSize: trainX[0].length});

                        // assigned updated weights to it's current index position relative to the layer's index
                        this.weights[pointer] = res1.params;

                        // assigned updated biases to it's current index position relative to the layer's index
                        this.biases[pointer] = res2.params;

                        // assigned updated weight states to it's current index position relative to the layer's index
                        this.optimizerStates.weights[pointer] = res1.state;

                        // assigned updated bias states to it's current index position relative to the layer's index
                        this.optimizerStates.biases[pointer] = res2.state;

                        pointer++;
                    }

                    this.#reinitiateWeightSBiasGrads(); // reset grads (weights and biases grads) to 0s

                    setGlobalParams(
                        this.weights, 
                        this.biases, 
                        this.output_layers_templates,
                    );

                }

                let AverageEpochLoss = totalepochLoss / numBatches; 
                let setColor = AverageEpochLoss > 0.9 ? color.red : 
                                AverageEpochLoss > 0.5 ? color.orange :
                                AverageEpochLoss > 0.1 ? color.yellow :
                                AverageEpochLoss > 0.03 ? color.lime : color.green;
                let end = performance.now();
                let totalDuration = (end - startTime) / 1000;

                previousEpochLoss = AverageEpochLoss;
                
                logMessage += `| [Epoch Loss]: ${setColor} ${AverageEpochLoss.toFixed(7)} ${color.reset}`;

                if (this.task === 'regression') {
                    let duration = `| [took: ${formatDuration(totalDuration)} to finish]`
                    logMessage += duration;
                }

                let accuracy = 0;
                let duration = "";

                if (this.task === 'binary_classification' || this.task === 'multi_class_classification') {
                    let epochPredictions = [];
                    for (let i = 0; i < trainX.length; i++) {
                        epochPredictions.push(this.#Feedforward(trainX[i]).predictions);
                    }

                    accuracy = this.#calculateClassificationAccuracy(epochPredictions, trainY, this.task);

                    let accuracyColor = accuracy > 90 ? color.green :
                                    accuracy > 85 ? color.lime :
                                    accuracy >= 75 ? color.yellow :
                                    accuracy >= 60 ? color.orange : color.red;

                    logMessage += ` | [Accuracy in Training]: ${accuracyColor} ${accuracy.toFixed(2)}% ${color.reset}`;
                    duration = `| [took: ${formatDuration(totalDuration)} to finish]`
                    logMessage += duration;
                }

                if (this.plugins?.trainingVisualizer) {

                    // pass data to the visualizer plugin and call the visualize() factory function. 
                    // Note: any visualizer plugins must expose a visualize() function and accepts an Object argument.
                    // The object contains data to be visualize in a moving graph for loss and accuracy (if needed).

                    const visualizerData = {
                        epoch: current_epoch + 1,
                        loss: AverageEpochLoss,
                        task: this.task,
                        totalEpoch: epoch,
                        totalBatchSize: batchSize,
                        optimizer: this.optimizer.name || "Custom Optimizer",
                        learningRate: this.learning_rate,
                        duration: duration,
                        version: version
                    }

                    if (this.task === 'binary_classification' || this.task === 'multi_class_classification') {
                        visualizerData.accuracy = accuracy;
                    }

                    // dispatch the data to the plugin visualizers
                    for (const visualizer of this.plugins.trainingVisualizer) {
                        if (typeof visualizer?.visualize !== 'function') {
                            this.isfailed = true;
                            throw new Error("Failed to use a visualizer plugin. No callable 'visualize()' function.");
                        }
                        await visualizer.visualize(visualizerData);
                    }
                    
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

            if (this.plugins?.trainingVisualizer) {
                for (const visualizer of this.plugins.trainingVisualizer) {
                    if (typeof visualizer?.abort !== 'function') {
                        this.isfailed = true;
                        throw new Error("Failed to use a visualizer plugin. No callable 'abort()' function.");
                    }

                    await visualizer.abort();
                }
            }
            
        }
        catch (error) {
            console.log(error);
            process.exit(1);
        }
    }

    /**
     *  @method predict()
        @param {Array} input - input data 
        @returns Array of predictions
        @throws Error when there's shape mismatch and no input data

     produces predictions based on the input data
    */
    async predict(input) {
        if (!this.isInit) {
            init();
            this.isInit = true;
        }

        setGlobalParams(this.weights, this.biases, this.output_layers_templates);

        if (!this.weights || !this.biases || !this.output_layers_templates) {
            throw new Error("Parameters are missing");
        }

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

                input[i] = input[i] instanceof Float32Array ? input[i] : new Float32Array(input[i].flat(Infinity));
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
            console.error(error);
        }
    }

    // ========= Private methods =======
    #build() {
        try {
            let [H, W, D] = this.input_shape;
            this.currentShape = [H, W, D];
            this.currentSize = H * W * D;

            this.layers.forEach((layer_data) => {
                
                const {
                    updatedSize, 
                    updatedShape, 
                    weights, 
                    biases, 
                    weightGrads, 
                    biasGrads, 
                    outputTensors, 
                    inputShape, 
                    outputShape, 
                    paramShape, isParametric, overrides } = layer_data.initParams(this.currentSize, this.currentShape, layer_data);

                this.currentSize = updatedSize;
                this.currentShape = updatedShape;

                if (weights.length > 0) this.weights.push(weights);
                if (biases.length > 0) this.biases.push(biases);
                if (weightGrads.length > 0) this.weightGrads.push(weightGrads);
                if (biasGrads.length > 0) this.biasGrads.push(biasGrads);
                if (outputTensors.length > 0) this.output_layers_templates.push(outputTensors);
                if (isParametric) this.parametric_layers.push(layer_data.layer_name);
                layer_data.weightShape = paramShape || [];
                layer_data.inputShape = inputShape || [];
                layer_data.outputShape = outputShape || [];
                if (overrides) {
                    this.input_shape = overrides.input_shape; // to override the default `this.input_shape` in the constructor just incase no layer.inputShape() was added to the sequentialBuild()
                    this.input_size = overrides.input_size; // to override the default `this.input_size` in the constructor just incase no layer.inputShape() was added to the sequentialBuild()
                }
            });

            this.hasBuilt = true;
        } catch (error) {
            console.error(`${color.red}[BUILD ERROR]------- ${error.message}${color.reset}`);
            throw error;
        }
    }

    // build single function is used when adding layer individually. Usually executes if add_layer() is called
    // it recieves layer data, same as `#build()` but instead of looping, it directly access the layer data being passed to
    // `add_layer()` and run the `initParams()` from the layer's configuration object
    #buildSingle(layer_data) {
        
        const {
            updatedSize, 
            updatedShape, 
            weights, 
            biases, 
            weightGrads, 
            biasGrads, 
            outputTensors, 
            inputShape, 
            outputShape, 
            isParametric,
            paramShape } = layer_data.initParams(this.currentSize, this.currentShape, layer_data);

        this.currentSize = updatedSize;
        this.currentShape = updatedShape;

        if (weights.length > 0) this.weights.push(weights);
        if (biases.length > 0) this.biases.push(biases);
        if (weightGrads.length > 0) this.weightGrads.push(weightGrads);
        if (biasGrads.length > 0) this.biasGrads.push(biasGrads);
        if (outputTensors.length > 0) this.output_layers_templates.push(outputTensors);
        if (isParametric) this.parametric_layers.push(layer_data.layer_name); 
        layer_data.weightShape = paramShape || [];
        layer_data.inputShape = inputShape || [];
        layer_data.outputShape = outputShape || [];
    }

    // backprop loop
    #backpropagation(activations, zs, deltas_array) {
        let deltas = deltas_array;
        let current_delta = deltas[this.num_layers - 1];
        let all_deltas = [current_delta];

        const layerPointers = [];
        let p = 0;
        const parametric = this.parametric_layers;
        for (let i = 0; i < this.layers.length; i++) {
            layerPointers.push(parametric.includes(this.layers[i].layer_name) ? p++ : -1);
        }

        for (let layer_index = this.num_layers - 2; layer_index >= 0; layer_index--) {
            const current_layer = this.layers[layer_index];
            const next_layer    = this.layers[layer_index + 1];

            const pointer = layerPointers[layer_index + 1];

            const dLda = next_layer.projectDeltaBackward(
                current_delta,
                pointer,
                current_layer.outputShape,
                next_layer
            );

            current_delta = current_layer.applyOwnDerivative(
                dLda,
                zs[layer_index],
                current_layer
            );

            deltas[layer_index] = current_delta;
            all_deltas.unshift(current_delta);
        }

        return { deltas, all_deltas };
    }


    // forward propagation
    #Feedforward(input) {
        let current_input = input
        let all_layer_outputs = [input];
        let zs = [];
        
        let outputTemplatePointer = 0
        let pointer = 0;
        for (let layer_index = 0; layer_index < this.num_layers; layer_index++) {
            const current_layer = this.layers[layer_index];
            // const layer_weights = this.weights[pointer];
            // const layer_biases = this.biases[pointer];

            const { outputs, z_values, incrementor_value } = current_layer.feedforward(current_input, current_layer, pointer, outputTemplatePointer);
            pointer+=incrementor_value;
            outputTemplatePointer++;

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

    /**
     * 
     * @param {Object} data layer data 
     * @param {Array<Float32Array>} weights array of weights
     * @param {Array<Float32Array>} biases array of biases
     * @param {String} fileName model filename 
     */
    #save(data, weights, biases, fileName) {
        if (this.isfailed) {
            console.log('[FAILED]------- Failed to save model');
        }
        else {
            const dir = process.cwd() //path.dirname(require.main.filename);

            // Record tensor lengths (in elements, not bytes) so we can slice the
            // binary block back into individual Float32Arrays on load.
            data.weightLengths = weights.map(w => w.length);
            data.biasLengths = biases.map(b => b.length);

            // Metadata (architecture, hyperparams, shapes) is small text data,
            // so JSON + deflate is fine here.
            const metaJson = Buffer.from(JSON.stringify(data), 'utf-8');
            const metaCompressed = zlib.deflateSync(metaJson);

            // Weights/biases are large numeric tensors. Never run these through
            // JSON.stringify: converting millions of floats to decimal text
            // inflates a compact 4-byte-per-value Float32Array into a string many
            // times larger, and at scale this exceeds V8's max string length
            // (~512MB), throwing "RangeError: Invalid string length".
            // Instead, view each Float32Array's underlying memory directly as a
            // Buffer (zero-copy) and concatenate into one raw binary block.
            const weightBuffers = weights.map(w => Buffer.from(w.buffer, w.byteOffset, w.byteLength));
            const biasBuffers = biases.map(b => Buffer.from(b.buffer, b.byteOffset, b.byteLength));
            const tensorBlock = Buffer.concat([...weightBuffers, ...biasBuffers]);
            const tensorCompressed = zlib.deflateSync(tensorBlock);

            // Define file format:
            // [HEADER (4 bytes)] + [VERSION (1 byte)] + [META_LENGTH (4 bytes, uint32 LE)]
            // + [META (compressed JSON)] + [TENSOR BLOCK (compressed raw floats)]
            const header = Buffer.from("NRX4"); // Magic bytes (bumped: new binary tensor format). Note: This is a breaking changes because. Loading models which are from the older version will cause an error because the loading model function will try to reconstruct how saved model function serialized the model
            const version = Buffer.from([0x04]); // Version 4

            const metaLengthBuf = Buffer.alloc(4);
            metaLengthBuf.writeUInt32LE(metaCompressed.length, 0);

            // Combine all parts
            const finalBuffer = Buffer.concat([header, version, metaLengthBuf, metaCompressed, tensorCompressed]);

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

    #recalculateShape() {
        let H = this.input_shape[0];
        let W = this.input_shape[1];
        let D = this.input_shape[2];
        let sequenceLength = 1;

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
            else if (layer.layer_name === "maxPooling") {
                const [poolHeight, poolWidth] = layer.poolSize;
                const stride = layer.strides;
                const padding = layer.padding;

                const {OutputHeight, OutputWidth} = calculateTensorShape(H, W, poolHeight, poolWidth, D, stride, padding);
                H = OutputHeight;
                W = OutputWidth;
            } 

            else if (layer.layer_name === "connected_layer") {
                H = 1;
                W = 1;
                D = layer.layer_size;
            }
            else if (layer.layer_name === "EmbeddingLayer") {
                H = 1;
                W = 1;
                D = layer.maxSequenceLength * layer.embeddingDim;
                sequenceLength = layer.maxSequenceLength;
            }
            else if (layer.layer_name === "recurrent_cell") {
                H = 1;
                W = 1;
                D = layer.return_sequence ? (layer.units * layer.maxSequenceLength) : layer.units;
            }
        }

        this.currentShape = [H, W, D];
        this.currentSize = H * W * D;
    }

   #reinitiateWeightSBiasGrads() {

        for (let i = 0; i < this.weights.length; i ++) {
            this.weightGrads[i].fill(0);
            this.biasGrads[i].fill(0);
        }

        for (const template of this.output_layers_templates) {
            template.fill(0);
        }
    }
}

module.exports = Neurex;
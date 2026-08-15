
/**
 * Neurex - a Trainable Neural Network Library for NodeJS | Author: Kark Angelo V. Pada
 * 
 * Copyright (c) all rights reserved
 * 
 * Licensed under the MIT License.
 * See LICENSE file in the project root for full license information.
 * @module neurex
 * 
 */
declare module 'neurex' {
    /**
    * CsvDataHandler is a utility tool for that allows you to extract and manipulate data from your .csv dataset.
    *
    * @class
    */
    export class CsvDataHandler {
        /**
        * Opens and reads the provided CSV file and maps its contents into an array of arrays.
        * The first row is treated as column names and stored separately.
        *
        * @method read_csv
        * @param {string} filename - The path to the CSV file.
        * @returns {Array<Array<string>>} An array of arrays representing the CSV data, with column names removed from the data array.
        * @throws {Error} If no file is provided, or if the file has an unsupported extension.
        * @example
        * const loader = new CsvDataHandler();
        * try {
        * const data = loader.read_csv('my_data.csv');
        * console.log(data); // [[value1, value2], [value3, value4]]
        * console.log(loader.columnNames); // ['header1', 'header2']
        * } catch (error) {
        * console.error(error.message);
        * }
        */
        read_csv(filename: string): void;

        /**
        * Converts all elements in every row of the provided data array to numerical values.
        * Ensure that all elements are numeric, otherwise, they will result in `NaN`.
        *
        * @method rowsToInt
        * @param {Array<Array<string>>} data - The extracted data from the CSV, where elements are strings.
        * @returns {Array<Array<number>>} An array of arrays with all elements converted to numbers.
        * @throws {Error} If no data is provided.
        * @example
        * const loader = new CsvDataHandler();
        * const stringData = [['1', '2'], ['3', '4']];
        * const numberData = loader.rowsToInt(stringData);
        * console.log(numberData); // [[1, 2], [3, 4]]
        */
        rowsToInt(data: any[]): number[][];

        /**
        * Selects a range of elements from each row of the provided array.
        *
        * @method getRowElements
        * @param {number} setRange - The number of elements to select from the beginning of each row.
        * @param {Array<Array<any>>} array - The data from which to extract elements.
        * @returns {Array<Array<any>>} An array of arrays containing the selected elements.
        * @throws {Error} If `setRange` is invalid or `array` is not provided.
        * @example
        * const loader = new CsvDataHandler();
        * const data = [[1, 2, 3], [4, 5, 6]];
        * const selected = loader.getRowElements(2, data);
        * console.log(selected); // [[1, 2], [4, 5]]
        */
        getRowElements(setRange: number, array: number[][]): number[];

        /**
        * Removes specified columns from the dataset and updates the column names.
        *
        * @method removeColumns
        * @param {string[]} fields - An array of column names to remove.
        * @param {Array<Array<any>>} data - The dataset from which to remove columns.
        * @returns {Array<Array<any>>} The modified dataset with the specified columns removed.
        * @throws {Error} If no fields are provided or data is missing, or if a specified column is not found.
        * @example
        * const loader = new CsvDataHandler();
        * loader.columnNames = ['A', 'B', 'C'];
        * const data = [['a1', 'b1', 'c1'], ['a2', 'b2', 'c2']];
        * const newData = loader.removeColumns(['B'], data);
        * console.log(newData); // [['a1', 'c1'], ['a2', 'c2']]
        * console.log(loader.columnNames); // ['A', 'C']
        */
        removeColumns(fields: any[], data: any[][]  ): void;

        /**
        * Extracts a column as a 1D array and removes that column from the dataset and column names.
        *
        * @method extractColumn
        * @param {string} columnName - The name of the column to extract.
        * @param {Array<Array<any>>} data - The dataset rows from which to extract the column.
        * @returns {Array<any>} A 1D array containing the extracted values.
        * @throws {Error} If `columnName` or `data` is missing, or if the specified column is not found.
        * @example
        * const loader = new CsvDataHandler();
        * loader.columnNames = ['A', 'B', 'C'];
        * const data = [['a1', 'b1', 'c1'], ['a2', 'b2', 'c2']];
        * const extracted = loader.extractColumn('B', data);
        * console.log(extracted); // ['b1', 'b2']
        * console.log(data); // [['a1', 'c1'], ['a2', 'c2']] (data is mutated)
        * console.log(loader.columnNames); // ['A', 'C']
        */
        extractColumn(columnName: string, data: any[][]): any[];

        /**
        * Normalizes the provided data using the specified method.
        * Available methods:
        * - 'MinMax': normalizes data using Min-Max scaling. (0-1 range)
        * 
        * @method normalize
        * @param {String} method - the normalization method to use.
        * @param {Array<Array<number>>} data - the data to be normalized.
        * @throws {Error} If no method or data is provided, or if the method is unsupported.
        * @returns {Array<Array<number>>} - normalized data.
        * @example
        * const loader = new CsvDataHandler();
        * const data = [[1, 2], [3, 4]];
        * const normalized = loader.normalize('MiMax', data);
        * console.log(normalized); // normalized data based on MinMax scaling;
        */
        normalize(method: string, data: number[][]): number[][];

        /**
        * 
        * Returns rows from row 1 to the specified range and removes the rest
        *
        * @method trimRows
        * @param {Number} range - range
        * @param {Array<Array<any>>} data - the extracted data
        * @returns {Array<Array<any>>} - trim dataset
        * @throws {Error} - if no parameters are passed
        *
        *
        *
        */
        trimRows(range: number, data: any[][]): any[][];

        /**
        * Displays the provided data in a tabular format, including column names.
        *
        * @method tabularize
        * @param {Array<Array<any>>} data - The data to display in a tabular format.
        * @throws {Error} If no data is provided.
        * @example
        * const loader = new CsvDataHandler();
        * loader.columnNames = ['Name', 'Age'];
        * const data = [['Alice', 30], ['Bob', 24]];
        * loader.tabularize(data);
        * // Expected output in console:
        * // Name    Age
        * // Alice   30
        * // Bob     24
        */
        tabularize(data: any[][]): void;

        /**
        * 
        * 
        * Export the loaded data to CSV.
        * @param {String} file_Name - name of your CSV file
        * @param {Array<Array<any>>} data
        *
        * 
        */
        exportCSV(file_Name: string, data: [][]): void;
    }

    /**
    * @class MinMaxScaler
    * Scales input features (array of arrays) to [0, 1] based on feature-wise min/max.
    * Requires fitting on training data first.
    */
    export class MinMaxScaler {
        /**
        * Calculates min and max for each feature from the input data.
        * @param {Array<Array<number>>} data - The training data (e.g., X_train).
        */
        fit(data: number[]): void;

        /**
        * Transforms the input data using the fitted min and max values.
        * @param {Array<Array<number>>} data - The data to transform (e.g., X_train, X_test).
        * @returns {Array<Array<number>>} The normalized data.
        */
        transform(data: number[]): number[];

        /**
        * Inverse transforms the normalized data back to original scale.
        * @param {Array<Array<number>>} data - The normalized data to inverse transform.
        * @returns {Array<Array<number>>} The data transformed back to original scale.
        */
        inverseTransform(data: number[]): number[];
    }

    export interface onChangeConfig {
        /** set the epoch when to change the optimizer automatically. */
        targetEpoch?: Number;
        /** Optimizer function to use. */
        optimizer?: (params: any) => Function;
    }

    export interface pluginFactoryObject {
        /** type of plugin. All plugins must have a type */
        type?: string;
        /** initialize the plugin visualizer. This can be a localhost web server, websocket server, etc. */
        initilaize?: () => void;
        /** broadcast data */
        visualize?: (...params: any) => void;
        /** Every visualizer plugins must have an functional `abort` function. The core engine might use this to stop any running services (like ocalhost web server, websocket server, etc) when an error occurred in the core itself.*/
        abort?: () => void;

        /** for upcoming plugins, it is encourage to understand how the core framework works under the hood and how built-in ones exposed their factory functions*/
    }

    export interface NeurexConfig {
        /** Learning rate for training. Default: 0.001 */
        learning_rate?: number;
        /** Optimizer function to use. */
        optimizer?: (params: any) => Function;
        /** Set a checkpoint per N epochs. Every N epochs will save the model. (example: if you enter 10, then every 10 epochs will save the model)*/
        checkpoint_per_epoch?: number;
        /** if set to true, it won't use the compiled binaries, but instead uses the JS modules */
        onFLoat32Module?: true | false;
        /** set mode to `cpu`, `gpu` or `auto`. Default is `cpu`*/
        mode?: "cpu" | "gpu" | "auto";
        /** Learning rate scheduler function (`stepDecay()`, `exponentialDecay(), cosineAnnealing(), reduceOnPlateau()`)*/
        lr_scheduler?: (params: any) => Function;
        /** A clip norm value is a maximum threshold limit used in machine learning to prevent exploding gradients by scaling down oversized gradient vectors. Default is `1.0`*/
        clip_norm_value?: Number;
        /** on change config to automate changing of optimizer mid-training.*/
        onChange_optimizer?: onChangeConfig;
        /** visualizer plugin array. Accepts plugin factory functions */
        visualizerPlugins?: pluginFactoryObject[];
    }

    /**
     *
     * The core class of the library.
     *
     *
     * @class Neurex
     * 
     */
    export class Neurex {
        /**
        * Allows configuration of your neural network's parameters.
        * @method configure
        * @param {NeurexConfig} configs - Configuration options for the neural network.
        *
        * You may configure them optionally. Be careful of tweaking them as they will have an effect on your model's performance.
        *
        */
        
        configure(configs: NeurexConfig): void;

        /**
        * 
        @method modelSummary()

        Shows the model architecture
        */
        modelSummary(): void;

        /**
        * Get the input shape. Works only after loading a model or after sequential building
        *
        * @returns tensor input shape as [H, W, C]
        */
        getTensorShape(): Number[];

        /**
        * Get the input size
        * 
        * @returns the input size equivalent of number of features as innput
        */
        getInputSize(): Number;

        /**
         * get the task type. Can be use to identify what model is trained on and what it is trained for
         * @returns the task type: regression | multi_class_classification | binary_classification
         */
        get_task_type(): String;

        /**
        * @method get_miscellaneous_data
        * @returns {any} Saved miscellaneous data upon model saving
        */
        get_miscellaneous_data(): Object;
        
        /**
        * 
        * `saveModel()` allows you to save your model's architecture, weights, and biases, as well as other parameters. The model will be exported
        *  as a .nrx (neurex) model
        * 
        * @method saveModel()
        * @param {string} modelName the filename of your model
        * @param {Object} miscellaneous data that can be included to be saved in the model. Note: This may increase the model file size when adding miscellaneous.
        *   
        */
        saveModel(modelName: string, miscellaneous: object): void;

        /**
        * @method loadSavedModel() method allows you to load the trained model. The model is typically in .nrx file format which contains the learned parameters of your trained model
        * @param {String} model the trained model file name
        * @param {Boolean} showLog outputs confirmation log when loading and successfullu loading a model. Default value is `true`.
        * @returns {void}
        */
        loadSavedModel(model: string, showLog: Boolean): void;

        /**
        * @method pop - Removes the last layer of the model including it's initialzed or trained parameters and optimizer states. Useful for transfer learning
        * @throws {Error} - if there are no layers
        */
        pop(): void;

        /**
        * @method add_layer - Appends a new layer to an existing model architecture. Upon appending a new layer will initiates untrained parameters.
        * @param {Object} layer_data - layer data returned from Layers class
        *
        * @example
        * // sample usage
        * nrx.add_layer(layer.connectedLayer("relu", 10));
        */
        add_layer(layer_data: Object): void;

        /**
        * 
        * @method sequentialBuild
        * 
        * interface to stack layer types. No weights and biases initialization here
        * @param {Object} layer_data
        */
        sequentialBuild(layer_data: any[]): void;

        /**
        * Trains the neural network using the provided training data, target values, number of epochs, and learning rate.
        * 
        * 
        * @method train()
        * @param {Array<Array<number>>} trainX - The input training data. Each element is an array representing a single sample's features.
        * @param {Array<number>} trainY - The target values (ground truth) corresponding to each sample in trainX.
        * @param {string} loss - loss function to use: MSE, MAE, binary_cross_entropy, categorical_cross_entropy, sparse_categorical_cross_entropy
        * @param {Number} epoch - the number of training iteration
        * @param {Number} batch_size - mini batch sizing
        * 
        * @throws {Error} Throws an error if any required parameter is missing.
        * @returns Progress of every epoch can be print in the console.
        * 
        * @example
        * // Example usage:
        * 
        * const {Neurex, Layers} = require('neurex');
        * const model = new Neurex();
        * const layer = new Layers();
        *
        * model.sequentialBuild([
        *    layer.inputShape({features: 4}),
        *    layer.connectedLayer("relu", 3),
        *    layer.connectedLayer("relu", 3),
        *    layer.connectedLayer("softmax", 2)
        * ]);
        *
        * model.train(X_train, Y_train, 'categorical_cross_entropy', 2000, 12);
        * 
        * 
        */
        train(trainX: number[][], trainY: number[], loss: string, epoch: number, batch_size: number): void;

        /**
        * 
        @method predict
        @param {Array} input - input data 
        @returns Array of predictions
        @throws Error when there's shape mismatch and no input data

        produces predictions based on the input data
        */
        predict(input: number[][]): number[];
    }

    /**
    * Splits a dataset into training and testing sets.
    * @async
    * @function split_dataset
    * @param {Array<Array<number>>} X - array of features (input data)
    * @param {Array<number>} Y - array of labels (target data)
    * @param {number} split_ratio - the ratio for the test set (e.g., 0.2 for 20%)
    * @returns {object} {X_train, Y_train, X_test, Y_test}
    */
    export function split_dataset(X: number[][], Y: number[], split_ratio: number): {
        X_train: any[][], 
        Y_train: any[][], 
        X_test: any[][], 
        Y_test: any[][]
    };

    /**
    * Computes evaluation metrics for regression tasks given test features and labels.
    *
    * @function RegressionMetrics
    * @param {Array<Array<number>>} predictions The input features for the test set.
    * @param {Array<number>} actuals The true target values for the test set.
    * @param {Boolean} showOutputs shows the outputs. You can disable it by passing a boolean value. Default is `true`
    * @throws {Error} when textX and testY are not provided
    */
    export function RegressionMetrics(predictions: number[][], actuals: number[], showOutputs: Boolean): void;

    /**
    *
    * Computes evaluation metrics for classification tasks given predicted values and true labels.
    *
    * @function ClassificationMetrics
    * @param {Array<Array<number>>} predictions The predicted class labels or probabilities for the test set.
    * @param {Array<Array<number>>} actuals The true target class labels for the test set.
    * @param {string} classificationType binary, categorical, or sparse_categorical
    * @param {Array<any>} labels add labels that represents a class
    * @param {Boolean} showOutputs shows the misclassified outputs. You can disable it by passing a boolean value. Default is `true`
    */
    export function ClassificationMetrics(predictions: number[][], actuals: number[][], classificationType: string, labels: any[], showOutputs: Boolean): void;

    /**
    * Converts a column of categorical labels into one-hot encoded vectors.
    * @async
    * @function OneHotEncoded
    * @param {Array<Array<any>>} data - An array where each inner array represents a row and contains a single categorical label.
    * @returns {Array<Array<Number>>} Returns One-hot encoded labels, suitable for categorical classification.
    * @throws {Error} - Throws an error if no data is provided, or if any row is not a single-element array.
    */
    export function OneHotEncoded(data: any[][]): number[][];

    /**
    * Converts labels that cannot be converted to interger labels (example: words). If your labels already integer-labeled (ex: 0, 1, 2, 3, ...), no need to use this function
    * @async
    * @function IntegerLabeling
    * @param {Array<Array<any>>} data - column of your dataset that can be use as categorical labeling 
    * @returns {Array<Array<Number>>} returns Intger-encoded labels. Which can be use for categorical classification, particularly when calculating sparse_categorical_cross_entropy
    * @throws {Error} - when no data is provided
    */
    export function IntegerLabeling(data: any[][]): number[][];

    /**
    * Converts labels that cannot be converted to binary labels (example: words). If your labels already 0s and 1s, no need to use this function
    * @async
    * @function BinaryLabeling
    * @param {Array<Array<any>>} data - column of your dataset that can be use as binary labeling (0 or 1)
    * @returns {Array<Array<Number>>} returns labels which contains 1 vector labels of 1s and 0s. Can be use for Binary classifcation
    * @throws {Error} - when no data is provided or there are more than two classes
    */
    export function BinaryLabeling(data: any[][]): number[][];

    /**
    * @async
    * @function load_images_from_directory
    * @param {String} targetDir target directory of your image datasets. The folders inside the target directory will represents as class names for the images inside. The first class being read will be the first class among all classes. Therefore, assign your data to it's correct class.
    * @param {Array<Number>} resize an array containing the values for resizing [H, W].
    * @param {String} pixelFormat grayscale, rgb, or rgba. "grayscale" - 1 channel, "rgb" - 3 channel, and "rgba" - 4 channels.
    * @param {String} label_mode specifies how the target labels are encoded and shaped. It lets you match your label format directly to your loss function. Mode: `binary`, `categorical`, `sparse`
    * @param {Number} limit_per_class limit the number of items per class
    * @returns {Object}
    */
    export function load_images_from_directory(targetDir: String, resize: number[], pixelFormat: String, label_mode: String, limit_per_class: number): { datasets: Array<Float32Array>, targetY: Array<Array<Number>>, labels: Array<Array<String>>, classes: Array<String>};

    /**
    * @async
    * @function load_single_image This function allows you to load a single image by specifying it's path
    * @param {String} file_path path to the image file (can be nested anywhere)
    * @param {Array<Number>} resize resize the image to [H, W]
    * @param {String} pixelFormat grayscale, rgb, or rgba.
    * @param {Boolean} showLog when set to `true`, it will show the output logs after an image is loaded. Default value is `false`
    * @returns {{datasets: Array<Float32Array>, shape: Array<Number>, filename: filename}}
    */
    export function load_single_image(file_path: String, resize: Number[], pixelFormat: String, showLog: Boolean): {datasets: Array<Float32Array>, shape: Array<Number>, filename: String};

    /**
    * @async
    * @function load_multiple_images allows you to load a multiple images at once by specifying the folder that contains images
    * @param {String} file_path path to the image file (can be nested anywhere)
    * @param {Array<Number>} resize resize the image to [H, W]
    * @param {String} pixelFormat grayscale, rgb, or rgba.
    * @returns {{datasets: Array<Float32Array>, paths: Array<String>, filenames: Array<String>}}
    */
    export function load_multiple_images(file_path: String, resize: Number[], pixelFormat: String): {datasets: Array<Float32Array>, paths: Array<String>, filenames: Array<String>};

    /**
     * @function tokenize allows you to tokenize a sentence
     * @param {String} sentence input sentence
     * @returns {Array<String>} array of tokenized words 
     */
    export function tokenize(sentence: String): String[];

    /**
     * @function buildVocab - allows you to tokenized an entire corpus into tokens of words, symbols, numbers and removing duplicated words.
     * @param sentences an array of sentences or large corpus
     * @returns {Array<String>} an array of tokenized words
     */
    export function buildVocab(sentences: Array<String>): Array<String>;

    /**
     * @function buildWord2Id - this function assign unique token IDs to tokenized words. These token IDs will be use to `Encode` input tokenized words.
     * @param {Array<String>} vocab tokenized words
     * @returns {Object} an object containing key value pairs. Each key (words) has corresponding value (token ID)
     */
    export function buildWord2Id(vocab: String[]): Object;

    /**
     * @function Encode tokenize a sentence and assign token IDs returning an array of token IDs.
     * @param {String} sentence input sentence or prompt
     * @param {Object} buildWord2Id_output the output after calling `buildWord2Id()` function. This key-value object will be use to encode the input sentence and assign corresponding token IDs based on the words in `buildWord2Id_output`
     * @param {Number} max_length The length of the encoded token containing token IDs.
     * @returns {Array<Number>} an array of token IDs to be use for token embeddings in the embedding layer 
     */
    export function Encode(sentence: String, buildWord2Id_output: Object, max_length: Number): Array<Number>;

    /**
     * The `Layers` class acts as a factory for generating neural network layer configurations.
     * * Instead of holding network state directly, it provides a suite of callable builder methods. 
     * Each method returns a structured configuration object containing the parameters, feedforward, 
     * and backpropagation logic for that specific layer type. 
     * * These configurations are designed to be stacked sequentially inside an array and passed 
     * directly to `sequentialBuild()` to construct your model architecture.
     *
     * @class
     */
    export class Layers {
        /**
        * @method inputShape
        * @param {Object} shapeConfig - specify the number of features
        *
        * The inputShape() method allows you to get the shape of your input
        * @example
        * model.sequentialBuild([
            layer.inputShape({features: 4}),
            layer.connectedLayer("relu", 5),
            layer.connectedLayer("softmax", 3);
        ]);
        */
        inputShape(shapeConfig: Object): Object;

        /**
         * @method reshape changes the dimensions (shape) of the data passing through it without changing the data values. This acts as the `input layer` to bridge data from layers that outputs 1D vector to be feed to convolutional layers which works on spatial grid-like data. 
         * @param targetShape specify the target shape for the data to be reshape. Default is `[28, 28, 3]`
         * @returns {Object} The reshape layer object configuration
         */
        reshape(targetShape: number[]): object;

        /**
        * @method embeddingLayer Creates an embedding layer for token encoding.
        * @param {Number} vocabSize The size of the vocabulary.
        * @param {Number} embeddingDim The size of the dense vector used to represent each token.
        * @param {Number} maxSequenceLength The length of the encoded token containing token IDs.
        * @returns {Object} The embedding layer object configuration
        */
        embeddingLayer(vocabSize: Number, embeddingDim: Number, maxSequenceLength: Number): Object;

        /**
        * @method connectedLayer Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will build connected layers or multilayer perceptron
        * @param {Number} layer_size specify the number of neuron for this layer. Default is `5`
        * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear, softmax). Default is `relu`.
        * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
        */
        connectedLayer(layer_size: number, activation: string): Object;

        /**
        * 
        * @method convolutionalLayer Allows you to add convolutional layers in your model architecture in sequential building.
        * @param {Number} filters the number of filters for this convolutional layer. Produces the same number of output features
        * @param {Number} strides It determines how much the filter overlaps with the input as it slides across.
        * @param {Array<Number>} kernel_size the size of the kernel (or filter) that will slide and extracts input features
        * @param {String} activation_function the activation function to be use for this layer
        * @param {String} padding adds extra values (typically 0s) around the border of an input before applying a convolutional filter
        * @returns {Object} The convolutional layer object configuration
        * @throws {Error} if any of the parameters are invalid.
        */
        convolutionalLayer(filters: Number, strides: Number, kernel_size: Number[], activation_function: String, padding: string): Object;

        /**
        * @method maxPooling is use for downsampling operation that reduces the spatial dimensions of an input tensor by taking the maximum value over a defined sliding window
        * @param {Array<Number>} poolSize determines the pool size window
        * @param {Number} strides It determines how much the pool window slides across the input tensor. Default is `1`
        * @param {String} padding `same` or `valid`. Default is `same`
        * @returns {Object} The max pooling layer configuration
        * @throws {Error} if any of the values are 0s or negative for the pool size and strides or the padding is invalid
        */
        maxPooling(poolSize: Number[], strides: Number, padding: String): Object;

        /**
         * @method `recurrentCell` is the fundamental building block of a Recurrent Neural Network (RNN) designed to process sequential data. It maintains an internal `memory` by taking its output from the previous time step and feeding it back into itself alongside the new input.
         * @param {Number} units This is the number of hidden units (neurons) in the layer. It dictates the dimensionality of the layer's output space and its internal memory state. 
         * @param {String} activation_function The activation function applied to the internal hidden state. Default value is `tanh`.
         * @param {Boolean} return_sequence default value is `false`. If `false`, Outputs only the final hidden state vector at the very last time step. If set to `true`, Outputs the hidden state vector for every single time step in the sequence. Must be set to `true` if another RNN layer follows.
         * @param {Boolean} return_state default value is `false`. If `true`, the layer will return its final hidden state vector as a separate tensor alongside its standard output.
         * @returns {Object} The Recurrent Cell configuration
         * @throws {Error} if internal initialization and computational process has an error.
         */
        recurrentCell(units: Number, activation_function: String, return_sequence: Boolean, return_state: Boolean): Object;

        /**
        * 
        * @method transConvLayer `transConv` (or transpose convolution) is a specialized convolutional layer that upsamples incoming tensor map, which does the opposite of the normal convolution
        * @param {Number} filters the number of filters for this convolutional layer. Produces the same number of output features
        * @param {Number} strides It determines how much the filter overlaps with the input as it slides across.
        * @param {Array<Number>} kernel_size the size of the kernel (or filter) that will slide and extracts input features
        * @param {String} activation_function the activation function to be use for this layer
        * @param {String} padding adds N amount of padding on all sides. Default is `same`
        * @param {Array<Number>} inputShape use to determine the shape of the input going to this layer, especially if the input comes from layers that works on 1D inputs (e.g. connected layers -> trans convolution where usual output shape of connected layers are [1, 1, outputSize])
        * @param {Boolean} useBias when set to `false`, the layer will not use bias and will skip bias initialization. Default value is `true`. 
        * @return {Object} transConv layer configs
        * @throws {Error} if any of the parameters are invalid.
        */
        transConvLayer(filters: Number, strides:Number, kernel_size: Number[], activation_function: String,  padding: String, inputShape: Number[], useBias: boolean): Object;
    }

    /**
     * 
     * Automate annotations with `Annotator` module
     *
     * @class Annotator
     */

    export interface AnnotatorConfig {
        /** Model file to be loaded */
        model_path?: String;
        /** Target directory of images */
        target_directory_path?: String;
        /** Array of class names */
        classes?: Array<String>;
        /** Load a CSV file */
        CSV_file_name?: String;
    }

    export class Annotator {
        /**
         * @method configure()
         * @param {Object} config - set configuration for annotation
         */
        configure(config: AnnotatorConfig): void;

        /**
         * @method init() - Initialize instances, loading the model and setting internal variables
         */
        init(): void;

        /**
         * @async
         * @method imageClassifier() - prepares data and internal variales for annotation
         */
        imageClassifier(): void;

        /**
         * @async
         * @method image_classify() - starts the annotation process. Automatically classify images and sorting them based on the predicted class. 
         */
        image_classify(): void;
    }

    // ================ Math Ops ======================= //

    /**
    * @function element_wise_mul use to multiply elements inside both arrays. Requires both arrays has same length;
    * @param {Array<Number>} flat_arr_1 a flat array input
    * @param {Array<Number>} flat_arr_2  a flat array input
    * @returns {Float32Array} A flat array output after multiplying input_array_1[i] to the values of input_array_2[i]
    * @throws an error will occured if both array are not equal in length
    */
    export function element_wise_mul(flat_arr_1: Number[], flat_arr_2: Number[]): Float32Array;


    /**
    * @function element_wise_sub use to subtract elements inside both arrays. Requires both arrays has same length;
    * @param {Array<Number>} flat_arr_1 a flat array input
    * @param {Array<Number>} flat_arr_2 a flat array input
    * @returns {Float32Array} A flat array output after subtracting input_array_1[i] to the values of input_array_2[i]
    * @throws an error will occured if both array are not equal in length
    */
    export function element_wise_sub(flat_arr_1: Number[], flat_arr_2: Number[]): Float32Array;

    /**
     * @function scaleDiff a function that takes 3 input arrays and perform subtraction of values from `arr1[i]` to `arr2[i]` then multiply to `arr3[i]`
     * @param arr1 a flat array input
     * @param arr2 a flat array input
     * @param arr3 a flat array input
     * @returns {Float32Array} A flat array output after performing `(arr1[i] - arr2[i]) * arr3[i]`
     * @throws an error will occured if both array are not equal in length
     */
    export function scaleDiff(arr1: Number[], arr2: Number[], arr3: Number[]): Float32Array;

    /**
     * @function relu
     * @param {Float32Array} arr Float32Array values
     * @returns {Float32Array} relu output
     *
     * ReLu (Rectified Linear Unit) is an activation function where all the values are passed the same and zeroed out negative values
     */
    export function relu(arr: Float32Array): Float32Array;

    /**
     * @function sigmoid
     * @param {Float32Array} arr Float32Array values
     * @returns {Float32Array} sigmoid output
     *
     * Sigmoid is an activation function that squashes all values between 0 to 1. Ideal for binary classificaton tasks
     */
    export function sigmoid(arr: Float32Array): Float32Array;

    /**
     * @function tanh
     * @param {Float32Array} arr Float32Array values
     * @returns {Float32Array} tanh output
     *
     * Tanh (hyperbolic tangent) is an activation function that squashes all values between -1 to 1. Ideal for binary classificaton tasks
     */
    export function tanh(arr: Float32Array): Float32Array;

    /**
     * @function softmax
     * @param {Float32Array} arr Float32Array values
     * @returns {Float32Array} softmax output
     *
     * The softmax function is a mathematical tool that converts a vector of raw, real-numbered scores (logits) into a probability distribution, with values between 0 and 1 that sum up to exactly 1.
     * This activation function is primarily use in output layer.
     */
    export function softmax(arr: Float32Array): Float32Array;

    /**
     * @function linear
     * @param {Float32Array} arr Float32Array values
     * @returns {Float32Array} linear output
     *
     * The linear activation function outputs the same inputs directly without non-linear transformation. This means that whateveer being passed here, the same will be the output.
     */
    export function linear(arr: Float32Array): Float32Array;

    /**
     * @function detectGPU() 
     
     * - Runs a quick detection test for GPU availability. This is also used internally for CPU/GPU branching
     * 
     * Example output:
     * ```bash
     *   {
     *       ok: true,
     *       error: '',
     *       platformCount: 1,
     *       devices: [
     *           {
     *               gpu: 'Intel(R) UHD Graphics',
     *               vendor: 'Intel(R) Corporation',
     *               platform: 'Intel(R) OpenCL Graphics',
     *               driverVersion: '32.0.101.6127',
     *               openclVersion: 'OpenCL 3.0 NEO ',
     *               deviceType: 'gpu',
     *               globalMemBytes: 3378651136n,
     *               computeUnits: 32,
     *               maxClockMHz: 1250,
     *               hostUnifiedMemory: true
     *           }
     *       ]
     *   }
     *```
     */
    export function detectGPU(): Object;

    /**
     * provides some predefined network templates which can be drop in the `sequentialBuild()`. The templates doesn't have input layer nor a predefined output layer so that you can add your own.
     * The templates returns an array of layer configuration objects. To add them in the `sequentialBuild()`, you must use a spread operator (`...`)
     *
     * @example
     * 
     * nrx.sequentialBuild([
     *      layer.inputShape({features: 4}),
     *      ...templates.simpleNeuralNetwork(),
     *      layer.connectedLayer('linear', 1),
     * ]);
     * 
    */
    export namespace templates {
        /**
         * A simple neural network having 3 hidden connected layers, having 5 neurons each layer. All uses `relu` activation function
         */
        export function simpleNeuralNetwork(): Array<Object>;
    
        /**
         *  A simple CNN having a two convolutional layers each having different number of filters, same strides and kernel sizes. Both uses `same` padding and `relu` activation functions.
         * After each convolutional layers comes with max pooling layer having `2x2` pool sizes, 2 strides and uses `valid` padding. Then it uses 3 connected layers having a
         * "funnel" shape architecture 
         *
         * @param {Boolean} isHeadless if set to `true`, it will only return the extractor layers (convolution and max pooling layers). Default value is `false`
         */
        export function simpleCNN(isHeadless: Boolean): Array<Object>;

        /**
         * 
         * A deep convolutional neural network model consisting of 16 layers. This template allows to train VGG16 without manually placing the layers piece by piece in the `sequentialBuild()`
         */
        export function VGG16(): Array<Object>; 

        /**
         * A lightweight, deep convolutional neural network model. This template allows you to use `LiteNet` architecture where you can drop in the `sequentialBuild()`
         */
        export function LiteNet():Array<Object>;

        /**
         *  A vanilla recurrent neural network with 3 recurrent cells.
         * @param {Number} units_per_cell number of units per recurrent cells. Default is `3` 
         * @param {String} activation_function activation function to be used by recurrent cells. Default is `tanh`
         * 
         *
         */
        export function vanillaRNN(units_per_cell: Number, activation_function: String): Array<Object>;

        /**
         * A type of neural network which has an decoding and encoding parts
         */
        export function AutoEncoder(): Array<Object>;
    }

    /**
     * @function stepDecay Reduces the learning rate by a fixed factor after a set number of epochs.
     * @param {Number} dropFactor A drop factor in a learning rate scheduler is the multiplier used to reduce the learning rate. Default is `0.5`
     * @param {Number} dropEvery dropEvery (or drop_every) is a custom parameter used in step-decay learning rate schedulers to define the number of epochs or steps that pass before the learning rate drops by a specific multi-factor value. Default is `10`.
     */
    export function stepDecay(dropFactor: Number, dropEvery: Number): Number;

    /**
     * @function exponentialDecay Multiplies the learning rate by a decay constant raised to the power of the epoch or step.
     * @param {Number} decayRate is a multiplier factor that scales down the learning rate at each step or epoch. Default is `0.96`
     */
    export function exponentialDecay(decayRate: Number): Number;

    /**
     * @function cosineAnnealing Follows the shape of a cosine function to lower the learning rate smoothly to a minimum value.
     * @param {Number} totalEpochs The total number of epochs or steps over which the learning rate should decay following a cosine schedule.
     * @param {Number} minLR The minimum learning rate to decay toward.
     */
    export function cosineAnnealing(totalEpochs: Number, minLR: Number): Number;


    export interface ReduceOnPlateauConfig {
        /** Reduces the learning rate by multiplying it by this value. Default value is 0.5 */
        factor?: Number;
        /**Counts the number of epochs to wait with no improvement in the monitored metric before making a reduction. Default value is `5`*/
        patience?: Number;
        /** Sets a lower bound on the learning rate so it does not drop below this specific value. Default value is `1e-6`*/
        minLR?: Number;
    }
    /**
     * @function reduceOnPlateau Monitors a validation metric (like loss) and lowers the learning rate only when progress stops.
     * @param {ReduceOnPlateauConfig} config 
     */
    export function reduceOnPlateau(config: ReduceOnPlateauConfig): Number;

    /**
     * @function SGD or `Stochastic Gradient Descent` a core machine learning algorithm that updates model weights using small data batches or single samples, controlled by a learning rate and optional momentum.
     * @param {Number} momentum This hyperparameter dictates how much of the past gradient step is carried over to the current update. Default value is `0.9`.
     */
    export function SGD(momentum: Number): Function;

    /**
     * @function Adam or `Adaptive Moment Estimation` optimizer is a popular algorithm used to train deep learning models. Note: tweaking this can heavily skew training behavior. 
     * @param {Number} beta1 The exponential decay rate for the moving average of past gradients (the first moment or mean). Default value is `0.9`.
     * @param {Number} beta2 The exponential decay rate for the moving average of squared past gradients (the second moment or uncentered variance). Default value is `0.999`.
     * @param {Number} epsilon  A tiny positive constant added to the denominator. Default value is `1e-8`.
     */
    export function Adam(beta1: Number, beta2: Number, epsilon: Number): Function;

    /**
     * @function lossVisualizer is built in application for visualizing training progress. Keep track of loss and accuracy (if present) in a moving graph.
     */
    export function lossVisualizer(): Object;


    export interface lossLandscapeOption {
        /** re-render the loss landscape every N epoch. */
        renderEveryTargetEpoch?: number;
    }

    /**
     * 
     * @param {Object} options config object
     */
    export function lossLandscapeVisualizer(options: lossLandscapeOption): Object;
    

    /**
     * a visualizer tool that visualize model architecture and parameters
     */
    export function modelVisualizer(): Object;
}
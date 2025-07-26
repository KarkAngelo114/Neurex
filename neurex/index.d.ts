
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
        * @param {Array<Array<number>} data - the data to be normalized.
        * @throws {Error} If no method or data is provided, or if the method is unsupported.
        * @returns {Array>Array<number>} - normalized data.
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
    * This class allows you to run inference predictions on your applications. You can load your trained model and run predictions
    *
    * @class
    */
    export class Interpreter {
        /**
        * @method loadSavedModel()
        * @param {*} model - the trained model

        The loadSavedModel() method allows you to load the trained model. The model is typically in .nrx file format which contains the learned parameters of your trained model

        */
        loadSavedModel(model: string): void;

        /**
        * 
        @method predict
        @param {Array} input - input data 
        @returns Array of predictions
        @throws Error when there's shape mismatch and no input data

        produces predictions based on the input data
        */
        predict(input: number[][]): number[];

        /**
        * 
        @method modelSummary

        Shows the model architecture
        */
        modelSummary(): void;
    }

    /**
    * @method MinMaxScaler
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
    export class Neurex {
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
        configure(configs: object): void;

        /**
        * @method inputShape()
        * @param {Array} data - the dataset

        the inputShape() method allows you to get the shape of your input.
        This will tell the network that your input layer has this X number of input neuron.
        Ensure that your dataset has no missing values, otherwise perform data cleaning.
        */
        inputShape(data: any[][]): void;

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
        flatten(input: number[][]): number[];

        /**
        * 
        @method modelSummary()

        Shows the model architecture
        */
        modelSummary(): void;

        /**
        * 

        @method construct_layer()
        @param {String} activation_func specify the activation function for this layer (Available: sigmoid, relu, tahn, linear)
        @param {Number} layer_size specify the number of neuron for this layer.

        The construct_layer() method allows you to create layers of your network. 
        Each layer has its own number of neurons and all uses the same activation function
        to output new features before passing to the next layer.

        */
        construct_layer(activation_func: string, layer_size: number): void;

        /**
        * 
        @method saveModel()
        @param {string} modelName - the filename of your model. If not provided, the filename of the model is date today.

        saveModel() allows you to save your model's architecture, weights, and biases, as well as other parameters. The model will be exported
        as a .nrx (neurex) model and a metadata.json will be generated along with the model file.
        
        */
        saveModel(modelName: string): void;

        /**
        * Trains the neural network using the provided training data, target values, number of epochs, and learning rate.
        * 
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
        * const nn = new Neurex();
        * nn.inputShape(trainX); // Set input shape based on your data
        * nn.construct_layer('relu', 8); // Add a hidden layer with 8 neurons and ReLU activation
        * nn.construct_layer('linear', 1); // Add an output layer with 1 neuron and linear activation
        * nn.train(X_train, Y_train, 'mse', 100, 4); // Train for 100 epochs and a loss function of 'mse' and a batch size of 4
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
    * @param {Array<Array<number>>} X - array of features (input data)
    * @param {Array<number>} Y - array of labels (target data)
    * @param {number} split_ratio - the ratio for the test set (e.g., 0.2 for 20%)
    * @returns {object} {X_train, Y_train, X_test, Y_test}
    */
    export function split_dataset(X: number[][], Y: number[], split_ratio: number): object;

    /**
    * Computes evaluation metrics for regression tasks given test features and labels.
    *
    * @function
    * @param {Array<Array<number>>} predictions - The input features for the test set.
    * @param {Array<number>} actuals - The true target values for the test set.
    * @throws {Error} when textX and testY are not provided
    */
    export function RegressionMetrics(predictions: number[][], actuals: number[]): void;

    /**
    *
    * Computes evaluation metrics for classification tasks given predicted values and true labels.
    *
    * @function ClassificationMetrics
    * @param {Array<Array<number>>} predictions - The predicted class labels or probabilities for the test set.
    * @param {Array<Array<number>>} actuals - The true target class labels for the test set.
    * @param {string} classificationType - binary, categorical, or sparse_categorical
    * @param {Array<any>} labels - (Optional) - add labels that represents a class
    */
    export function ClassificationMetrics(predictions: number[][], actuals: number[][], classificationType: string, labels: any[]): void;

    /**
    * Converts a column of categorical labels into one-hot encoded vectors.
    *
    * @param {Array<Array<any>>} data - An array where each inner array represents a row and contains a single categorical label.
    * @returns {Array<Array<Number>>} Returns One-hot encoded labels, suitable for categorical classification.
    * @throws {Error} - Throws an error if no data is provided, or if any row is not a single-element array.
 */
    export function OneHotEncoded(data: any[][]): number[][];

    /**
    * Converts labels that cannot be converted to interger labels (example: words). If your labels already integer-labeled (ex: 0, 1, 2, 3, ...), no need to use this function
    * @param {Array<Array<any>>} data - column of your dataset that can be use as categorical labeling 
    * @returns {Array<Array<Number>>} returns Intger-encoded labels. Which can be use for categorical classification, particularly when calculating sparse_categorical_cross_entropy
    * @throws {Error} - when no data is provided
    */
    export function IntegerLabeling(data: any[][]): number[][];

    /**
    * Converts labels that cannot be converted to binary labels (example: words). If your labels already 0s and 1s, no need to use this function
    * @param {Array<Array<any>>} data - column of your dataset that can be use as binary labeling (0 or 1)
    * @returns {Array<Array<Number>>} returns labels which contains 1 vector labels of 1s and 0s. Can be use for Binary classifcation
    * @throws {Error} - when no data is provided or there are more than two classes
    */
    export function BinaryLabeling(data: any[][]): number[][];
}
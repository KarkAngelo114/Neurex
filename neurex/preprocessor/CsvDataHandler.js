/**
 *
 * CsvDataHandler
 *
 */

const fs = require('fs');
const path = require('path');
const {MinMaxScaler} = require('./normalizer');

/**
 * CsvDataHandler is a utility tool for that allows you to extract and manipulate data from your .csv dataset.
 *
 * @class
 */
class CsvDataHandler {
    /**
     * Creates an instance of CsvDataHandler.
     */
    constructor() {
        /**
         * The expected file extension for CSV files.
         * @private
         * @type {string}
         */
        this._FILE_EXTENSION = '.csv';
        /**
         * The name of the loaded file.
         * @type {string}
         */
        this.fileName = '';
        /**
         * Array of column names extracted from the CSV file.
         * @type {string[]}
         */
        this.columnNames = [];
        /**
         * The raw data extracted from the CSV file, as an array of arrays.
         * @type {Array<Array<string>>}
         */
        this.data = [];
    }

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
    read_csv(filename) {
        if (!filename) {
            throw new Error("[ERROR]------ No file provided.");
        }

        const dir = path.dirname(require.main.filename);
        const extension_name = path.extname(filename);

        if (extension_name !== this._FILE_EXTENSION) {
            throw new Error(`[ERROR]------- Unsupported file extension '${extension_name}'. Only accepts '${this._FILE_EXTENSION}' format.`);
        }

        this.fileName = filename;
        const csvPath = path.join(dir, filename);

        try {
            const fileContent = fs.readFileSync(csvPath, 'utf-8');
            const lines = fileContent.split('\n').filter(line => line.trim());

            if (lines.length === 0) {
                this.columnNames = [];
                this.data = [];
                return [];
            }

            // Extract column names from the first line
            this.columnNames = lines[0].split(',').map(cell => cell.trim());

            // Extract data rows (excluding the header)
            this.data = lines.slice(1).map(row => row.split(',').map(cell => cell.trim()));

            return this.data;
        } catch (err) {
            // Re-throw specific errors or wrap them for better context
            if (err.code === 'ENOENT') {
                throw new Error(`[ERROR]------- File not found: ${csvPath}`);
            }
            throw new Error(`[ERROR]------- Failed to read CSV file: ${err.message}`);
        }
    }

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
    rowsToInt(data) {
        if (!data) {
            throw new Error("[ERROR]------- No data is passed.");
        }

        return data.map(arr => {
            return arr.map(cell => Number(cell));
        });
    }

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
    getRowElements(setRange, array) {
        if (typeof setRange !== 'number' || isNaN(setRange) || setRange < 0 || !array) {
            throw new Error(`[ERROR]------- Invalid setRange: ${setRange} or array: ${array}.`);
        }

        return array.map(arr => {
            return arr.slice(0, setRange);
        });
    }

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
    removeColumns(fields = [], data) {
        if (!Array.isArray(fields) || fields.length === 0 || !data) {
            throw new Error(`[ERROR]------- Invalid fields: ${fields} or data: ${data}.`);
        }

        let currentColumnNames = [...this.columnNames];
        let currentData = data.map(row => [...row]); // Create a shallow copy of data to avoid direct mutation issues during index shifting

        // Identify indices to remove, and sort them in descending order
        // to avoid issues with index shifting when splicing
        const indicesToRemove = fields
            .map(field => {
                const index = currentColumnNames.indexOf(field);
                if (index === -1) {
                    throw new Error(`[ERROR]------- Column '${field}' not found.`);
                }
                return index;
            })
            .sort((a, b) => b - a); // Sort descending

        // Remove columns from data
        for (const rowIndex in currentData) {
            for (const index of indicesToRemove) {
                currentData[rowIndex].splice(index, 1);
            }
        }

        // Remove column names
        for (const index of indicesToRemove) {
            currentColumnNames.splice(index, 1);
        }

        this.columnNames = currentColumnNames;
        return currentData;
    }

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
    extractColumn(columnName, data) {
        if (!columnName || !data) {
            throw new Error(`[ERROR]------- columnName: ${columnName}, data: ${data}.`);
        }

        const index = this.columnNames.indexOf(columnName);
        if (index === -1) {
            throw new Error(`[ERROR]------- Column '${columnName}' not found.`);
        }

        // Remove the column name from the internal array
        this.columnNames.splice(index, 1);

        // Extract column values and simultaneously modify rows
        const extractedValues = data.map(row => {
            const value = row[index];
            row.splice(index, 1); // Mutate the row to remove the value
            return [value];
        });

        return extractedValues;
    }


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

    normalize(method, data) {
        try {
            // check if method and data are provided
            if (!method || !data || data.length == 0) {
                throw new Error(`[ERROR]------- No method nor data is provided.`);
            }

            // perform normalization based on the provided method
            if (method.toLowerCase() ==="minmax") {
                const scaler = new MinMaxScaler(); // using the MinMaxScaler class from the nornamizer module
                scaler.fit(data); // fit data to get the min and max values
                return scaler.transform(data); // returns the normalized data;
            }
            else {
                throw new Error(`[ERROR]------- Unsupported normalization`);
            }
        }
        catch (error) {
            console.log(error);
        }
    }

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
    trimRows(range, data) {
        try { 
            return data.slice(0, range);
        }
        catch (error) {
            console.log(error);
        }
    }

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
    tabularize(data) {
        if (!data || data.length === 0) {
            throw new Error(`[ERROR]-------- No data is provided.`);
        }

        if (Array.isArray(data[0])) {
            // Calculate maximum column widths for proper alignment
            const columnWidths = this.columnNames.map(name => name.length);
            data.forEach(row => {
                row.forEach((cell, i) => {
                    columnWidths[i] = Math.max(columnWidths[i], String(cell).length);
                });
            });

            // Print column names
            let headerRow = '';
            this.columnNames.forEach((name, i) => {
                headerRow += name.padEnd(columnWidths[i] + 4); // Add padding for spacing
            });
            console.log(headerRow);
            console.log('-'.repeat(headerRow.length)); // Separator line

            // Print data rows
            data.forEach(row => {
                let rowData = '';
                row.forEach((cell, i) => {
                    rowData += String(cell).padEnd(columnWidths[i] + 4);
                });
                console.log(rowData);
            });
        }
        else {
            data.forEach((row, i) => {
                console.log(i+1,'.',row);
            });
        }
    }

    /**
     * 
     * @param {String} file_Name - name of your CSV file
     * @param {Array<Array<any>>} data
     *
     * Export the loaded data to CSV.
     */
    exportCSV(file_Name, data) {
        const columnNames = this.columnNames;
        const dir = path.dirname(require.main.filename);
        const escape = (value) => {
            if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
            return `"${value.replace(/"/g, '""')}"`; // escape double quotes
            }
            return value;
        };

        const headers = columnNames.join(',');
        const NewData = data.map(row => row.map(escape).join(',')).join('\n');

        const csv = `${headers}\n${NewData}`;

        const file = path.join(dir, `${file_Name}.csv`);
        fs.writeFileSync(file, csv);

        console.log(`[SUCCESS]------- Exported file exported as ${file_Name}.csv`);

    }
}

module.exports = CsvDataHandler;
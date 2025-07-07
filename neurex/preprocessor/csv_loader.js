/**
 * 

csv_reader

 */

const fs = require('fs');
const path = require('path');

/**
 * 

csv_loader is a utility tool for Neurex that allows you to extract data from your .csv dataset  


@class
 */
class csv_loader {
    constructor() {
        this.file_extension = '.csv';
        this.file = '';
        this.column_names = [];
        this.data;
    }


    /**
     * 
     * opens and reads the provided CSV file and map the contents into an array. This will include the column names and the rows.
     * To remove the column names, you can pass a boolean parameter. If set to true, it will remove the array of columns before returning
     * the data, otherwise, it will be return along with the extracted data. Column names can still be use by getColumnData() for referencing
     *
     * @method read_csv()
     * @param {string} filename - the CSV file
     * @param {boolean} removeColumnNames -  If set to true, it will remove the array of columns before returning
     * the data, otherwise, it will be return along with the extracted data.
     * @returns array of arrays
     * @throws {Error} - if the file extension is not a CSV file or there is no provided file
     */

    read_csv(filename, removeColumnNames = false) {
        try {
            if (!filename) {
                throw new Error("[ERROR]------ No file provided");
            }
            const dir = path.dirname(require.main.filename);
            const extension_name = path.extname(filename);

            if (extension_name !== this.file_extension) {
                throw new Error(`[ERROR]------- Unsupported file extension '${extension_name}'. Only accepts '.csv' format. `);
            }

            const csv = path.join(dir, filename);

            let data = fs.readFileSync(csv, 'utf-8').split('\n').filter(line => line.trim()).map(row => row.split(',').map(cell => cell.trim()));
            data[0].forEach(col => {
                this.column_names.push(col);
            });
            
            if (removeColumnNames) {
                data.splice(0, 1);
                this.data = data;
                return data;
            }
            else {
                this.data = data;
                return data;
            }
            
        }
        catch (err) {
            console.error(err);
        }
    }


    /**
     * 

     @param {Array<Array<any>>} - extracted data from the CSV
     @returns - rows are now intergers
     @throws {Error} - if data is not present


     When extracting data from a CSV, all rows are strings. Passing the extracted data in this method will transform all elements in every row to numerical
     values. Ensure that no elements is non-numeric, otherwise, it will result to NaN.
     */
    RowsToInt(data) {
        try {
            if (!data) {
                throw new Error("[ERROR]------- No data is passed");
            }

            return data.map(arr => {
                return arr.map(row => {
                    return Number(row);
                });
            });
        }
        catch (err) {
            console.error(err);
        }
    }

    /**
     * 
     * @param {number} setRange - select the range of elements in a row 
     * @param {Array<number>} array - extracted data from the CSV   
     * @returns - selected elements
     */

    getRowElements(setRange, array) {
        try {
            if (!setRange || isNaN(setRange || !array)) {
                throw new Error(`[ERROR]------- setRange: ${setRange}`);
            }

            return array.map(arr => {
                return arr.slice(0, setRange);
            });
        }
        catch (err) {
            console.error(err);
        }
    }

    /**
     * @method removeColumn
     * @param {String} column_name - the colmun name
     * @param {Array<array>} data - extracted data from the CSV
     * @returns array of the data under the column
     *
     * Delete an entire column
     */
    removeColumn(column_name, data) {
        try {
            if (column_name === '' || column_name == undefined || column_name == null) {
                throw new Error(`[ERROR]------- column_name: ${column_name}, data: ${data}`);
            }

            const index = this.column_names.indexOf(column_name);

            // Check if the column exists
            if (index === -1) {
                throw new Error(`[ERROR]------- Column '${column_name}' not found.`);
            }

            this.column_names.splice(index, 1);

            return data.map(array => {
                // Create a new array excluding the element at 'index'
                const newArray = [...array]; // Create a shallow copy
                newArray.splice(index, 1); // Remove 1 element at 'index'
                return newArray;
            });
        } catch (err) {
            console.error(err);
        }
    }


    /**
    * @method extractColumn
    * Extracts a column as a 1D array while also removing that column from the dataset and column names.
    *
    * @param {string} column_name - The name of the column to extract
    * @param {Array<Array<any>>} data - The dataset rows
    * @returns {Array<any>} - The extracted values as a 1D array
    */
    extractColumn(column_name, data) {
        try {
            if (!column_name || !data) {
                throw new Error(`[ERROR]------- column_name: ${column_name}, data: ${data}`);
            }

            const index = this.column_names.indexOf(column_name);
            if (index === -1) {
                throw new Error(`[ERROR]------- Column '${column_name}' not found.`);
            }

            // Remove the column name
            this.column_names.splice(index, 1);

            // Extract column values and simultaneously modify rows
            const extracted_values = data.map(row => {
                const value = row[index];
                row.splice(index, 1); // Mutate the row to remove the value
                return value;
            });

            return extracted_values;
        } catch (err) {
            console.error(err);
        }
    }

}


module.exports = csv_loader;
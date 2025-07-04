// /**
//  * @method minMaxZeroOne
//  * Scales input array to [0, 1]
//  */
// const MinMaxScaler = (inputArr) => {
//     let normalized_array = [];
//     inputArr.forEach(inputArr => {
//         const min = Math.min(...inputArr);
//         const max = Math.max(...inputArr);
//         normalized_array.push(inputArr.map(x => (x - min) / (max - min)));
//     });
//     return normalized_array;
// };

// /**
//  * @method minMaxCustomRange
//  * Scales input array to [a, b]
//  */
// const minMaxCustomRange = (inputArr, a, b) => {
//     const min = Math.min(...inputArr);
//     const max = Math.max(...inputArr);
//     return inputArr.map(x => ((x - min) / (max - min)) * (b - a) + a);
// };

// /**
//  * @method minMaxManual
//  * Scales input array using provided min and max
//  */
// const minMaxManual = (inputArr, min, max) => {
//     return inputArr.map(x => (x - min) / (max - min));
// };

// module.exports = {
//     MinMaxScaler,
//     minMaxCustomRange,
//     minMaxManual
// };



/**
 * @method MinMaxScaler
 * Scales input features (array of arrays) to [0, 1] based on feature-wise min/max.
 * Requires fitting on training data first.
 */
class FeatureMinMaxScaler {
    constructor() {
        this.min_vals = null;
        this.max_vals = null;
    }

    /**
     * Calculates min and max for each feature from the input data.
     * @param {Array<Array<number>>} data - The training data (e.g., X_train).
     */
    fit(data) {
        if (data.length === 0) {
            throw new Error("Input data for scaler cannot be empty.");
        }
        const numFeatures = data[0].length;
        this.min_vals = Array(numFeatures).fill(Infinity);
        this.max_vals = Array(numFeatures).fill(-Infinity);

        data.forEach(row => {
            row.forEach((value, featureIdx) => {
                if (value < this.min_vals[featureIdx]) {
                    this.min_vals[featureIdx] = value;
                }
                if (value > this.max_vals[featureIdx]) {
                    this.max_vals[featureIdx] = value;
                }
            });
        });
    }

    /**
     * Transforms the input data using the fitted min and max values.
     * @param {Array<Array<number>>} data - The data to transform (e.g., X_train, X_test).
     * @returns {Array<Array<number>>} The normalized data.
     */
    transform(data) {
        if (!this.min_vals || !this.max_vals) {
            throw new Error("Scaler has not been fitted yet. Call fit() first.");
        }
        if (data.length === 0) return [];

        const normalized_data = [];
        data.forEach(row => {
            const normalized_row = [];
            row.forEach((value, featureIdx) => {
                const min = this.min_vals[featureIdx];
                const max = this.max_vals[featureIdx];
                if (max - min === 0) { // Handle case where feature has no variance
                    normalized_row.push(0); // Or original value, depending on desired behavior
                } else {
                    normalized_row.push((value - min) / (max - min));
                }
            });
            normalized_data.push(normalized_row);
        });
        return normalized_data;
    }
}

module.exports = {
    MinMaxScaler: FeatureMinMaxScaler, // Export the class
};
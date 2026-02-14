



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
        if (Array.isArray(data[0])) {
            // Normal 2D case
            const numFeatures = data[0].length;
            this.min_vals = Array(numFeatures).fill(Infinity);
            this.max_vals = Array(numFeatures).fill(-Infinity);

            data.forEach(row => {
                row.forEach((value, featureIdx) => {
                    if (value < this.min_vals[featureIdx]) this.min_vals[featureIdx] = value;
                    if (value > this.max_vals[featureIdx]) this.max_vals[featureIdx] = value;
                });
            });
        } else {
            // 1D case
            this.min_vals = [Math.min(...data)];
            this.max_vals = [Math.max(...data)];
        }

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
        if (Array.isArray(data[0])) {
            // Normal 2D
            return data.map(row => row.map((value, i) => {
                const min = this.min_vals[i];
                const max = this.max_vals[i];
                return max - min === 0 ? 0 : (value - min) / (max - min);
            }));
        } else {
            // 1D case
            const min = this.min_vals[0];
            const max = this.max_vals[0];
            return data.map(value => max - min === 0 ? 0 : (value - min) / (max - min));
        }

    }

    /**
     * Inverse transforms the normalized data back to original scale.
     * @param {Array<Array<number>>} data - The normalized data to inverse transform.
     * @returns {Array<Array<number>>} The data transformed back to original scale.
     */

    inverseTransform(data) {
        if (Array.isArray(data[0])) {
            // 2D array
            return data.map(row => row.map((value, i) => 
                value * (this.max_vals[i] - this.min_vals[i]) + this.min_vals[i]
            ));
        } else {
            // 1D array
            const min = this.min_vals[0];
            const max = this.max_vals[0];
            return data.map(value => value * (max - min) + min);
        }
    }


    /**
     * @method getMinMax
     * @returns {Array<number} - Returns the min and max values for each feature.
     * Returns the min and max values.
     */
    getMinMax() {
        if (!this.min_vals || !this.max_vals) {
            throw new Error("Scaler has not been fitted yet. Call fit() first.");
        }
        return { min: this.min_vals, max: this.max_vals };
    }
}

module.exports = {
    MinMaxScaler: FeatureMinMaxScaler,
};
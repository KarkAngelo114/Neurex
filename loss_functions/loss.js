// loss.js
// loss functions used for training, updating weights and biases
// tells how "wrong" the network is in it's predictions vs actuals during training


// loss.js

const mse = (predictions, actual) => {
    if (Array.isArray(predictions) && Array.isArray(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            const p = predictions[i];
            const a = actual[i];
            if (Array.isArray(p) && Array.isArray(a)) { 
                // Handle batch of multi-output samples
                let innerSum = 0;
                for (let j = 0; j < p.length; j++) {
                    innerSum += Math.pow(p[j] - a[j], 2);
                }
                sum += innerSum / p.length;
            } else {
                sum += Math.pow(p - a, 2);
            }
        }
        return sum / predictions.length;
    } else if (Array.isArray(predictions) && !Array.isArray(actual)) {
        // predictions is vector, actual is scalar repeated
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            sum += Math.pow(predictions[i] - actual, 2);
        }
        return sum / predictions.length;
    } else {
        return Math.pow(predictions - actual, 2);
    }
};


const mae = (predictions, actual) => {
    if (Array.isArray(predictions) && Array.isArray(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            const p = predictions[i];
            const a = actual[i];
            if (Array.isArray(p) && Array.isArray(a)) { 
                // Handle batch of multi-output samples
                let innerSum = 0;
                for (let j = 0; j < p.length; j++) {
                    innerSum += Math.abs(p[j] - a[j]);
                }
                sum += innerSum / p.length;
            } else {
                sum += Math.abs(p - a);
            }
        }
        return sum / predictions.length;
    } else if (Array.isArray(predictions) && !Array.isArray(actual)) {
        // predictions is vector, actual is scalar repeated
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            sum += Math.abs(predictions[i] - actual);
        }
        return sum / predictions.length;
    } else {
        return Math.abs(predictions - actual);
    }
};


/**
 * Categorical Cross-Entropy Loss
 * @param {Array<number>} predictions - predicted probabilities (output of softmax)
 * @param {Array<number>} actual - one-hot encoded true labels
 * @returns {number} loss value
 */
const categorical_cross_entropy = (predictions, actual) => {
    // Add small epsilon to avoid log(0)
    const epsilon = 1e-15;
    return -actual.reduce((sum, y, i) => sum + y * Math.log(Math.max(predictions[i], epsilon)), 0);
};

/**
 * Sparse Categorical Cross-Entropy Loss
 * @param {Array<number>} predictions - predicted probabilities (output of softmax)
 * @param {number} actual - true class index (integer)
 * @returns {number} loss value
 */
const sparse_categorical_cross_entropy = (predictions, actual) => {
    const epsilon = 1e-15;
    const p = Math.max(predictions[actual], epsilon);
    return -Math.log(p);
};

/**
 * Binary Cross-Entropy Loss
 * @param {number|Array<number>} predictions - predicted probability/probabilities (output of sigmoid)
 * @param {number|Array<number>} actual - true label(s) (0 or 1)
 * @returns {number} loss value
 */
const binary_cross_entropy = (predictions, actual) => {
    const epsilon = 1e-15;
    // Support both single value and array
    if (Array.isArray(predictions) && Array.isArray(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            const p = Math.max(Math.min(predictions[i], 1 - epsilon), epsilon);
            sum += -(actual[i] * Math.log(p) + (1 - actual[i]) * Math.log(1 - p));
        }
        return sum / predictions.length;
    } else {
        const p = Math.max(Math.min(predictions, 1 - epsilon), epsilon);
        return -(actual * Math.log(p) + (1 - actual) * Math.log(1 - p));
    }
};

module.exports = {
    mse,
    mae,
    categorical_cross_entropy,
    sparse_categorical_cross_entropy,
    binary_cross_entropy
};
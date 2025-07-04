/**
 * Splits a dataset into training and testing sets.
 * @param {Array<Array<number>>} X - array of features (input data)
 * @param {Array<number>} Y - array of labels (target data)
 * @param {number} split_ratio - the ratio for the test set (e.g., 0.2 for 20%)
 * @returns {object} {X_train, Y_train, X_test, Y_test}
 */
const split_dataset = (X, Y, split_ratio) => {
    // Ensure X and Y have the same number of samples
    if (X.length !== Y.length) {
        throw new Error("X and Y must have the same number of samples.");
    }

    // Create an array of indices to shuffle X and Y together
    const indices = Array.from({length: X.length}, (_, i) => i);

    // Shuffle the indices randomly
    // A standard way to shuffle an array for sorting is using Math.random() - 0.5
    const shuffledIndices = [...indices].sort(() => Math.random() - 0.5); 

    const test_size = Math.floor(X.length * split_ratio);

    const X_train = [];
    const Y_train = [];
    const X_test = [];
    const Y_test = [];

    for (let i = 0; i < X.length; i++) {
        const original_index = shuffledIndices[i];
        if (i < test_size) {
            X_test.push(X[original_index]);
            Y_test.push(Y[original_index]);
        } else {
            X_train.push(X[original_index]);
            Y_train.push(Y[original_index]);
        }
    }

    return {X_train, Y_train, X_test, Y_test};
};

module.exports = split_dataset;
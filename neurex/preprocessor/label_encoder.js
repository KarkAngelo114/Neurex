

/**
 * Converts a column of categorical labels into one-hot encoded vectors.
 *
 * @param {Array<Array<any>>} data - An array where each inner array represents a row and contains a single categorical label.
 * @returns {Array<Array<Number>>} Returns One-hot encoded labels, suitable for categorical classification.
 * @throws {Error} - Throws an error if no data is provided, or if any row is not a single-element array.
 */
const OneHotEncoded = (data) => {
    if (!data) {
        throw new Error("[ERROR]------- No data is provided");
    }

    // Extract the actual labels from the nested arrays and validate them
    let extractedLabels = [];
    data.forEach((row, i) => {
        // Each row must be an array and contain exactly one element
        if (!Array.isArray(row) || row.length !== 1) {
            throw new Error(`[ERROR]------ Row at index ${i} must be an array containing exactly one element.`);
        }
        extractedLabels.push(row[0]); // Extract the label from the single-element array
    });

    // Create a set of unique labels to determine the number of classes
    const uniqueLabels = [...new Set(extractedLabels)];
    const numClasses = uniqueLabels.length;

    // Create a map from each unique label to its corresponding index
    const labelMap = new Map();
    uniqueLabels.forEach((label, index) => {
        labelMap.set(label, index);
    });

    // Generate the one-hot encoded vectors
    return extractedLabels.map(label => {
        const oneHotVector = Array(numClasses).fill(0);
        const index = labelMap.get(label);
        if (index === undefined) {
            throw new Error(`[ERROR]------ Label '${label}' not found in unique labels map.`);
        }
        oneHotVector[index] = 1;
        return oneHotVector;
    });
};

/**
 * Converts labels that cannot be converted to interger labels (example: words). If your labels already integer-labeled (ex: 0, 1, 2, 3, ...), no need to use this function
 * @param {Array<Array<any>>} data - column of your dataset that can be use as categorical labeling 
 * @returns {Array<Array<Number>} returns Intger-encoded labels. Which can be use for categorical classification, particularly when calculating sparse_categorical_cross_entropy
 * @throws {Error} - when no data is provided
 */
const IntegerLabeling = (data) => {

    if (!data) {
        throw new Error("[ERROR]------- no data is provided");
    }
    // rows must be an array and must contains 1 element
    data.forEach((row, i) => {
        if (!Array.isArray(row) && row[i].length != 1) throw new Error("[ERROR]------ Rows must be an Array and contains atleast 1 element")
    });

    let labels = [];
    data.forEach(label => {
        labels.push(label[0]);
    });

    const uniqueLabels = [...new Set(labels)];
    const labelMap = new Map();
    uniqueLabels.forEach((label, index) => {
        labelMap.set(label, index);
    });
    return labels.map(label => [labelMap.get(label)]);
};

/**
 * Converts labels that cannot be converted to binary labels (example: words). If your labels already 0s and 1s, no need to use this function
 * @param {Array<Array<any>>} data - column of your dataset that can be use as binary labeling (0 or 1)
 * @returns {Array<Array<Number>>} returns labels which contains 1 vector labels of 1s and 0s. Can be use for Binary classifcation
 * @throws {Error} - when no data is provided or there are more than two classes
 */

const BinaryLabeling = (data) => {
    if (!data || !Array.isArray(data)) {
        throw new Error("[ERROR] No data provided or data is not an array.");
    }

    // Validate each row
    data.forEach((row, i) => {
        if (!Array.isArray(row) || row.length !== 1) {
            throw new Error(`[ERROR] Row at index ${i} must be a single-element array.`);
        }
    });

    // Extract unique labels (converted to lowercase for consistency)
    let uniqueLabels = [];
    data.forEach(row => {
        const label = row[0].toLowerCase();
        if (!uniqueLabels.includes(label)) {
            uniqueLabels.push(label);
        }
    });

    if (uniqueLabels.length !== 2) {
        throw new Error("[ERROR] There must be exactly two classes for binary labeling.");
    }

    // Map labels to binary values
    const labelMap = {
        [uniqueLabels[0]]: 0,
        [uniqueLabels[1]]: 1
    };

    // Convert to binary labels
    const binaryData = data.map(row => [labelMap[row[0].toLowerCase()]]);

    return binaryData;
};


module.exports = {
    OneHotEncoded,
    IntegerLabeling,
    BinaryLabeling
}
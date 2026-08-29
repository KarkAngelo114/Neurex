/**
 * Calculates the average absolute value (L1 Mean) of a parameter array.
 * @param {Array<number>|TypedArray} params 
 * @returns {number} L1 Mean Value
 */
function L1_Mean(params) {

    if (!params || !params.length) return 0;

    // Flatten nested arrays if params are multi-dimensional
    const flatParams = Array.isArray(params) ? params.flat(Infinity) : params;

    const sum = flatParams.reduce((acc, val) => acc + Math.abs(val), 0);

    return parseFloat(sum / flatParams.length).toFixed(4);
}
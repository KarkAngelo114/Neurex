

const MSE = (predictions, actual) => {
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


const MAE = (predictions, actual) => {
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

const r2 = (predictions, actual) => {
    // Ensure both inputs are arrays (even if they are 1D arrays for single-output scenarios,
    // they should still be treated as such by checking Array.isArray)
    if (!Array.isArray(predictions) || !Array.isArray(actual)) {
        console.error("r2 function expects array inputs for both predictions and actual.");
        return NaN; // Or throw an error, depending on desired strictness
    }

    let totalValues = 0;
    let sumActual = 0;
    const flatActual = []; // To store all actual values as a 1D array
    const flatPredictions = []; // To store all prediction values as a 1D array

    // First, flatten both predictions and actuals into 1D arrays
    // This makes the subsequent R2 calculation simpler and more robust
    // for both single-output (e.g., [[val], [val]]) and multi-output (e.g., [[val1, val2], [val3, val4]])
    for (let i = 0; i < actual.length; i++) {
        // Handle cases where the inner element might be a single number (from a flattened 1D array passed as 2D)
        // or an array (for true 2D structure)
        if (Array.isArray(actual[i])) {
            if (actual[i].length !== predictions[i]?.length) {
                console.warn(`Row ${i} has different lengths in actual (${actual[i].length}) and predictions (${predictions[i]?.length}). Calculations might be inaccurate.`);
                // Decide how to handle this: skip row, return NaN, etc.
                // For R2, it's critical that predictions and actuals align.
                return NaN;
            }
            for (let j = 0; j < actual[i].length; j++) {
                flatActual.push(actual[i][j]);
                flatPredictions.push(predictions[i][j]); // Push corresponding prediction
                totalValues++;
                sumActual += actual[i][j];
            }
        } else {
            // This case handles if a "row" is just a number, which shouldn't happen
            // if it's strictly "array of arrays". This is a safeguard.
            console.error("Mixed dimensions in actual array. Expected array of arrays.");
            return NaN;
        }
    }

    if (totalValues === 0) {
        console.warn("Actual array is empty or contains no numeric values after processing. Cannot calculate R2.");
        return NaN;
    }

    const mean = sumActual / totalValues;
    let sum_total_sq = 0; // Sum of squared differences from the mean of actual values
    let sum_res_sq = 0;   // Sum of squared residuals (differences between actual and predictions)

    for (let i = 0; i < flatActual.length; i++) {
        sum_res_sq += Math.pow(flatActual[i] - flatPredictions[i], 2);
        sum_total_sq += Math.pow(flatActual[i] - mean, 2);
    }

    if (sum_total_sq === 0) {
        return sum_res_sq === 0 ? 1 : NaN;
    }

    return 1 - (sum_res_sq / sum_total_sq);
};

const rMSE = (predictions, actual) => {
    if (Array.isArray(predictions) && Array.isArray(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            let preds = predictions[i];
            let acts = actual[i];
            for (let j = 0; j < preds.length; j++) {
                sum += Math.pow(preds[j] - acts[j], 2);
            }
            
        }
        return Math.sqrt(sum / predictions.length);
    }
    else {
        return Math.abs(predictions - actual); // RMSE for single value is abs error
    }
}



module.exports = {
    MSE,
    MAE,
    r2,
    rMSE,
};
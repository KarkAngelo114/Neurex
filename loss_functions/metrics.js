

const isArrayLike = (value) => Array.isArray(value) || ArrayBuffer.isView(value);

const MSE = (predictions, actual) => {
    if (isArrayLike(predictions) && isArrayLike(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            const p = predictions[i];
            const a = actual[i];
            if (isArrayLike(p) && isArrayLike(a)) {
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
    } else if (isArrayLike(predictions) && !isArrayLike(actual)) {
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
    if (isArrayLike(predictions) && isArrayLike(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            const p = predictions[i];
            const a = actual[i];
            if (isArrayLike(p) && isArrayLike(a)) {
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
    } else if (isArrayLike(predictions) && !isArrayLike(actual)) {
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
    if (!isArrayLike(predictions) || !isArrayLike(actual)) {
        console.error("r2 function expects array inputs for both predictions and actual.");
        return NaN;
    }

    let totalValues = 0;
    let sumActual = 0;
    const flatActual = [];
    const flatPredictions = [];

    for (let i = 0; i < actual.length; i++) {
        const actualRow = actual[i];
        const predRow = predictions[i];

        if (isArrayLike(actualRow)) {
            if (!isArrayLike(predRow) || actualRow.length !== predRow.length) {
                console.warn(`Row ${i} has different lengths in actual (${actualRow.length}) and predictions (${predRow?.length}). Calculations might be inaccurate.`);
                return NaN;
            }
            for (let j = 0; j < actualRow.length; j++) {
                flatActual.push(actualRow[j]);
                flatPredictions.push(predRow[j]);
                totalValues++;
                sumActual += actualRow[j];
            }
        } else if (typeof actualRow === 'number') {
            if (isArrayLike(predRow)) {
                console.warn(`Row ${i} has mixed dimensions between actual and predictions.`);
                return NaN;
            }
            flatActual.push(actualRow);
            flatPredictions.push(predRow);
            totalValues++;
            sumActual += actualRow;
        } else {
            console.error("Mixed dimensions in actual array. Expected array of arrays.");
            return NaN;
        }
    }

    if (totalValues === 0) {
        console.warn("Actual array is empty or contains no numeric values after processing. Cannot calculate R2.");
        return NaN;
    }

    const mean = sumActual / totalValues;
    let sum_total_sq = 0;
    let sum_res_sq = 0;

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
    if (isArrayLike(predictions) && isArrayLike(actual)) {
        let sum = 0;
        let count = 0;
        for (let i = 0; i < predictions.length; i++) {
            const preds = predictions[i];
            const acts = actual[i];
            if (isArrayLike(preds) && isArrayLike(acts)) {
                for (let j = 0; j < preds.length; j++) {
                    sum += Math.pow(preds[j] - acts[j], 2);
                }
                count += preds.length;
            } else {
                sum += Math.pow(preds - acts, 2);
                count += 1;
            }
        }
        return count === 0 ? NaN : Math.sqrt(sum / count);
    } else {
        return Math.abs(predictions - actual);
    }
};

module.exports = {
    MSE,
    MAE,
    r2,
    rMSE,
};
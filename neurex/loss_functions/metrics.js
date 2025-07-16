

const MSE = (predictions, actual) => {
    if (Array.isArray(predictions) && Array.isArray(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            sum += Math.pow(predictions[i] - actual[i], 2);
        }
        return sum / predictions.length;
    }
    else {
        return Math.pow(predictions - actual, 2);
    }
};

const MAE = (predictions, actual) => {
    if (Array.isArray(predictions) && Array.isArray(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            sum += Math.abs(predictions[i] - actual[i]);
        }
        return sum / predictions.length;
    } 
    else {
        return Math.abs(predictions - actual);
    }
};

const r2 = (predictions, actual) => {
    if (Array.isArray(predictions) && Array.isArray(actual)) {
        const mean = actual.reduce((a, b) => a + b, 0) / actual.length;
        let sum_total_sq = 0, sum_res_sq = 0;
        for (let i = 0; i < predictions.length; i++) {
            sum_res_sq += Math.pow(actual[i] - predictions[i], 2);
            sum_total_sq += Math.pow(actual[i] - mean, 2);
        }
        return 1 - (sum_res_sq / sum_total_sq);
    }
    else {
        return predictions === actual ? 1 : 0;
    }
};

const rMSE = (predictions, actual) => {
    if (Array.isArray(predictions) && Array.isArray(actual)) {
        let sum = 0;
        for (let i = 0; i < predictions.length; i++) {
            sum += Math.pow(predictions[i] - actual[i], 2);
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
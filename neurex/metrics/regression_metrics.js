

const {MSE, MAE, r2, rMSE} = require('../loss_functions/metrics');

/**
 * Computes evaluation metrics for regression tasks given test features and labels.
 *
 * @function
 * @param {Array<Array<number>>} predictions - The input features for the test set.
 * @param {Array<number>} actuals - The true target values for the test set.
 * @throws {Error} when textX and testY are not provided
 */
const RegressionMetrics = (predictions, actuals) => {
    try  {
        if (!predictions || !actuals) {
            throw new Error("[ERROR]------- No predictions or testY is present");
        }

        if (predictions.length != actuals.length) {
            throw new Error(`[ERROR]------- Length mismatch: Number of predictions: ${predictions.length} | Number of Actual targets: ${actuals.length}`);
        }

        // do a for loop to check for input shape
        for (let i = 0; i < predictions.length; i++) {
            let preds = predictions[i];
            let acts = actuals[i];
            if (preds.length != acts.length) {
                throw new Error(`[ERROR]------- Shape mismatch: On row ${i}, the number of elements on predictions is ${preds.length} and the number of elements on actuals is ${acts.length} `);
            }
        }

        // do a for loop to check for NaNs either in predictions or actuals
        for (let i = 0; i < predictions.length; i++) {
            let preds = predictions[i];
            let acts = actuals[i]
            for (let j = 0; j < preds.length; j++) {
                if (isNaN(preds[j] || isNaN(acts[j]))) {
                    throw new Error(`[ERROR]------- Non-numeric values detected in row ${i}.`);
                }
            }
        }

        console.log("\nOutput:");
        // Print predictions and actuals side by side
        for (let i = 0; i < predictions.length; i++) {
            console.log("Predicted:",predictions[i],"| Actual:",actuals[i]);
        }
        
        console.log("\n======= Evaluation Metrics =======");
        console.log('MSE: ', MSE(predictions, actuals));
        console.log('MAE: ', MAE(predictions, actuals));
        console.log('r2: ', r2(predictions, actuals));
        console.log('rMSE: ', rMSE(predictions, actuals));
    }
    catch (error) {
        console.error(error)
    }
}

module.exports = RegressionMetrics;
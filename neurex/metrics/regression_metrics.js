

const {MSE, MAE, r2, rMSE} = require('../loss_functions/metrics');

/**
 * Computes evaluation metrics for regression tasks given test features and labels.
 *
 * @function
 * @param {Array<Array<number>>} predictions - The input features for the test set.
 * @param {Array<number>} testY - The true target values for the test set.
 * @throws {Error} when textX and testY are not provided
 */
const RegressionMetrics = (predictions, testY) => {
    try  {
        if (!predictions || !testY) {
            throw new Error("[ERROR]------- No predictions or testY is present");
        }

        // Flatten testY if it's an array of arrays
        const flat_test_Y = Array.isArray(testY[0]) ? testY.map(arr => arr[0]) : testY;

        // Check for length mismatch
        if (predictions.length !== flat_test_Y.length) {
            console.warn(`[WARNING] Length mismatch: predictions (${predictions.length}) vs actuals (${flat_test_Y.length})`);
        }

        // Check for non-numeric values
        const hasInvalid = predictions.some(x => typeof x !== 'number' || isNaN(x)) || flat_test_Y.some(y => typeof y !== 'number' || isNaN(y));
        if (hasInvalid) {
            console.warn('[WARNING] Non-numeric value detected in predictions or actuals.');
        }

        // Print predictions and actuals side by side
        console.log('\nPredictions\tActual');
        for (let i = 0; i < Math.max(predictions.length, flat_test_Y.length); i++) {
            console.log(`${predictions[i].toFixed(3)}\t\t${flat_test_Y[i].toFixed(3)}`);
        }
        
        console.log("\n======= Evaluation Metrics =======");
        console.log('MSE: ', MSE(predictions, flat_test_Y));
        console.log('MAE: ', MAE(predictions, flat_test_Y));
        console.log('r2: ', r2(predictions, flat_test_Y));
        console.log('rMSE: ', rMSE(predictions, flat_test_Y));
    }
    catch (error) {
        console.error(error)
    }
}

module.exports = RegressionMetrics;
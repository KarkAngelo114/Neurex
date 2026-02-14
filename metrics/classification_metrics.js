/**
 * Evaluates classification metrics on the provided test data.
 *
 */

/**
 *
 * Computes evaluation metrics for classification tasks given predicted values and true labels.
 *
 * @function ClassificationMetrics
 * @param {Array<Array<number>>} predictions - The predicted class labels or probabilities for the test set.
 * @param {Array<Array<number>>} actuals - The true target class labels for the test set.
 * @param {string} classificationType - binary, categorical, or sparse_categorical
 * @param {Array<any>} labels - (Optional) - add labels that represents a class
 */
const ClassificationMetrics = (predictions, actuals, classificationType = undefined, labels = []) => {
    try {
        if (!predictions || predictions.length === 0 || !actuals || actuals.length === 0) {
            throw new Error("[ERROR]------- No predictions or actuals provided.");
        }

        if (predictions.length !== actuals.length) {
            throw new Error(`[ERROR]------- Mismatch: ${predictions.length} predictions vs ${actuals.length} actuals.`);
        }

        if (!classificationType) {
            throw new Error(`[ERROR]------- Classification type is undefined or not provided.`);
        }

        let preprocessedPredictions = [];
        let trueLabels = [];
        let classes = [];

        classificationType = classificationType.toLowerCase();

        if (classificationType === "binary") {
            trueLabels = actuals.flat();
            predictions.forEach(pred => preprocessedPredictions.push(pred[0] >= 0.5 ? 1 : 0));
            classes = labels.length ? labels : [...new Set(trueLabels.sort())];
        } 
        else if (classificationType === "categorical") {
            // One-hot actuals → indices
            trueLabels = actuals.map(a => a.indexOf(1));
            predictions.forEach(pred => preprocessedPredictions.push(pred.indexOf(Math.max(...pred))));
            const numClasses = Math.max(...trueLabels.concat(preprocessedPredictions)) + 1;
            classes = labels.length ? labels : Array.from({ length: numClasses }, (_, i) => i);
        } 
        else if (classificationType === "sparse_categorical") {
            trueLabels = actuals.flat(); // already integer indices
            predictions.forEach(pred => preprocessedPredictions.push(pred.indexOf(Math.max(...pred))));
            const numClasses = Math.max(...trueLabels.concat(preprocessedPredictions)) + 1;
            classes = labels.length ? labels : Array.from({ length: numClasses }, (_, i) => i);
        } 
        else {
            throw new Error(`[ERROR]------- Unknown classification type: ${classificationType}`);
        }

        // Accuracy and report calculations (same as before)
        let correctCount = 0, misclassifiedCount = 0, misclassifiedExamples = [];

        for (let i = 0; i < trueLabels.length; i++) {
            if (preprocessedPredictions[i] === trueLabels[i]) {
                correctCount++;
            } else {
                misclassifiedCount++;
                misclassifiedExamples.push(
                    `Predicted: ${preprocessedPredictions[i]} | Actual: ${trueLabels[i]} | ` +
                    `Predicted Class: ${classes[preprocessedPredictions[i]]} | Actual Class: ${classes[trueLabels[i]]}`
                );
            }
        }

        const accuracy = correctCount / trueLabels.length;
        console.log(`\nAccuracy: ${(accuracy * 100).toFixed(2)}%`);
        console.log("Correct:", correctCount);
        console.log("Misclassified:", misclassifiedCount);

        if (misclassifiedExamples.length > 0) {
            console.log("\nMisclassified Examples:");
            misclassifiedExamples.forEach(x => console.log(x));
        }

        console.log('\n======= Classification Report =======');
        console.log('\t           precision recall f1-score support\n');

        const precisions = [], recalls = [], f1s = [], supports = [];

        for (let k = 0; k < classes.length; k++) {
            let TP = 0, FP = 0, FN = 0;

            for (let i = 0; i < trueLabels.length; i++) {
                if (preprocessedPredictions[i] === k && trueLabels[i] === k) TP++;
                if (preprocessedPredictions[i] === k && trueLabels[i] !== k) FP++;
                if (preprocessedPredictions[i] !== k && trueLabels[i] === k) FN++;
            }

            supports[k] = trueLabels.filter(label => label === k).length;
            const precision = (TP + FP === 0) ? 0 : TP / (TP + FP);
            const recall = (TP + FN === 0) ? 0 : TP / (TP + FN);
            const f1 = (precision + recall === 0) ? 0 : 2 * (precision * recall) / (precision + recall);

            precisions[k] = precision;
            recalls[k] = recall;
            f1s[k] = f1;

            console.log(`${classes[k]}\t\t\t${precision.toFixed(2)}\t${recall.toFixed(2)}\t${f1.toFixed(2)}\t${supports[k]}`);
        }

        const totalSupport = supports.reduce((a, b) => a + b, 0);
        const macroPrecision = precisions.reduce((a, b) => a + b) / classes.length;
        const macroRecall = recalls.reduce((a, b) => a + b) / classes.length;
        const macroF1 = f1s.reduce((a, b) => a + b) / classes.length;

        let weightedPrecision = 0, weightedRecall = 0, weightedF1 = 0;
        for (let k = 0; k < classes.length; k++) {
            weightedPrecision += precisions[k] * supports[k];
            weightedRecall += recalls[k] * supports[k];
            weightedF1 += f1s[k] * supports[k];
        }
        weightedPrecision /= totalSupport;
        weightedRecall /= totalSupport;
        weightedF1 /= totalSupport;

        console.log(`\naccuracy          \t\t\t${accuracy.toFixed(2)}\t${totalSupport}`);
        console.log(`macro avg               ${macroPrecision.toFixed(2)}\t${macroRecall.toFixed(2)}\t${macroF1.toFixed(2)}\t${totalSupport}`);
        console.log(`weighted avg            ${weightedPrecision.toFixed(2)}\t${weightedRecall.toFixed(2)}\t${weightedF1.toFixed(2)}\t${totalSupport}`);
    } catch (error) {
        console.error(error);
    }
};

module.exports = ClassificationMetrics;

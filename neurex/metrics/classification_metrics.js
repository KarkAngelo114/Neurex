/**
 * Evaluates classification metrics on the provided test data.
 *
 * @module classification_metrics
 */

/**
 *
 * Computes evaluation metrics for classification tasks given predicted values and true labels.
 *
 * @function
 * @param {Array} predictions - The predicted class labels or probabilities for the test set.
 * @param {Array} trueLabels - The true target class labels for the test set.
 * @param {Array.<string>} classes - An array of strings representing the class labels (e.g., ["class1", "class2"]).
 */
const evaluate = (predictions, trueLabels, classes) => {

    if (classes.length === 0) {
        throw new Error("[FAILED]-------Evaluation failed. Please provide classes");
    }

    // Ensure predictions are in the correct format (integer labels)
    let processedPredictions = [];
    if (classes.length > 2) {
        // Multi-class: if predictions are probabilities (arrays), convert to class index
        if (Array.isArray(predictions[0])) {
            processedPredictions = predictions.map(array => array.indexOf(Math.max(...array)));
        } else {
            // Already integer labels
            processedPredictions = predictions;
        }
    } else {
        // Binary class: convert probabilities to 0 or 1
        processedPredictions = predictions.map(pred => pred > 0.5 ? 1 : 0);
    }

    let misclassifiedExamples = [];
    let correctCount = 0;
    let misclassifiedCount = 0;

    for (let i = 0; i < trueLabels.length; i++) {
        if (processedPredictions[i] === trueLabels[i]) {
            correctCount++;
        } else {
            misclassifiedCount++;
            misclassifiedExamples.push(`Predicted: ${processedPredictions[i]} | Actual: ${trueLabels[i]} | Predicted Class: ${classes[processedPredictions[i]]} | Actual Class: ${classes[trueLabels[i]]}`);
        }
    }

    const accuracy = correctCount / trueLabels.length;

    console.log(`\nAccuracy:`, (accuracy * 100).toFixed(2) + "%");
    console.log("Correct:", correctCount);
    console.log("Misclassified:", misclassifiedCount);

    if (misclassifiedExamples.length > 0) {
        console.log("\nMisclassified Examples:");
        misclassifiedExamples.forEach(x => {
            console.log(x);
        });
    }

    console.log('\n======= Classification Report =======');
    console.log('\t           precision recall f1-score support\n');

    let precisions = new Array(classes.length).fill(0);
    let recalls = new Array(classes.length).fill(0);
    let f1s = new Array(classes.length).fill(0);
    let supports = new Array(classes.length).fill(0);

    for (let k = 0; k < classes.length; k++) {
        const currentClassLabel = k; // The numerical label for the current class
        const className = classes[k];

        let TP = 0, FP = 0, FN = 0;

        for (let i = 0; i < trueLabels.length; i++) {
            if (processedPredictions[i] === currentClassLabel && trueLabels[i] === currentClassLabel) {
                TP++;
            }
            if (processedPredictions[i] === currentClassLabel && trueLabels[i] !== currentClassLabel) {
                FP++;
            }
            if (processedPredictions[i] !== currentClassLabel && trueLabels[i] === currentClassLabel) {
                FN++;
            }
            if (trueLabels[i] === currentClassLabel) {
                supports[k]++;
            }
        }

        const precision = (TP + FP === 0) ? 0 : TP / (TP + FP);
        const recall = (TP + FN === 0) ? 0 : TP / (TP + FN);
        const f1 = (precision + recall === 0) ? 0 : 2 * (precision * recall) / (precision + recall);

        precisions[k] = precision;
        recalls[k] = recall;
        f1s[k] = f1;

        console.log(`${className}\t\t\t${precision.toFixed(2)}\t${recall.toFixed(2)}\t${f1.toFixed(2)}\t${supports[k]}`);
    }

    const totalSupport = supports.reduce((sum, current) => sum + current, 0);

    const macroPrecision = precisions.reduce((a, b) => a + b) / classes.length;
    const macroRecall = recalls.reduce((a, b) => a + b) / classes.length;
    const macroF1 = f1s.reduce((a, b) => a + b) / classes.length;

    let weightedPrecision = 0;
    let weightedRecall = 0;
    let weightedF1 = 0;

    for (let k = 0; k < classes.length; k++) {
        weightedPrecision += precisions[k] * supports[k];
        weightedRecall += recalls[k] * supports[k];
        weightedF1 += f1s[k] * supports[k];
    }

    weightedPrecision = totalSupport === 0 ? 0 : weightedPrecision / totalSupport;
    weightedRecall = totalSupport === 0 ? 0 : weightedRecall / totalSupport;
    weightedF1 = totalSupport === 0 ? 0 : weightedF1 / totalSupport;


    console.log(`\naccuracy          \t\t\t${accuracy.toFixed(2)}\t${totalSupport}`);
    console.log(`macro avg               ${macroPrecision.toFixed(2)}\t${macroRecall.toFixed(2)}\t${macroF1.toFixed(2)}\t${totalSupport}`);
    console.log(`weighted avg            ${weightedPrecision.toFixed(2)}\t${weightedRecall.toFixed(2)}\t${weightedF1.toFixed(2)}\t${totalSupport}`);
};

module.exports = evaluate;

/**
 * use for evaluating model's performance
 *
 * @module metrics
 */
declare module 'metrics' {
    /**
    * Computes evaluation metrics for regression tasks given test features and labels.
    *
    * @function
    * @param {Array<Array<number>>} predictions - The input features for the test set.
    * @param {Array<number>} testY - The true target values for the test set.
    * @throws {Error} when textX and testY are not provided
    */
    export function RegressionMetrics(predictions: number[][], testY: number[]): void;
}
/**
 * Compute weight gradients (returns a 2D array)
 */
function computeWeightGradients(onGPU, a_prev, delta) {
    const gradients = [];
    for (let i = 0; i < a_prev.length; i++) {
        const row = [];
        for (let j = 0; j < delta.length; j++) {
            row.push(a_prev[i] * delta[j]);
        }
        gradients.push(row);
    }
    return gradients;
}


/**
 * Scale gradients (1D or 2D)
 */
function scaleGradients(onGPU, grad, scalar) {
    if (Array.isArray(grad[0])) {
        // 2D array
        return grad.map(row => row.map(v => v / scalar));
    } else {
        // 1D array
        return grad.map(v => v / scalar);
    }
}

module.exports = {
    computeWeightGradients,
    scaleGradients,
};

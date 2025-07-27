const { GPU } = require('gpu.js');
const gpu = new GPU({mode:'gpu'});

/**
 * Outer product kernel: ∇W = a_prev ⊗ delta
 * inputSize = length of a_prev
 * outputSize = length of delta
 */
const weightGradientKernel = (inputSize, outputSize) => {
    return gpu.createKernel(function(a_prev, delta) {
        return a_prev[this.thread.y] * delta[this.thread.x];
    }).setOutput([outputSize, inputSize]); // [columns][rows]
};

/**
 * Scale gradient by scalar (e.g., divide by batch size)
 */
const scaleKernel = (size) => {
    return gpu.createKernel(function(grad, scalar) {
        return grad[this.thread.x] / scalar;
    }).setOutput([size]);
};

/**
 * Compute weight gradients (returns a 2D array)
 */
function computeWeightGradients(onGPU, a_prev, delta) {
    if (onGPU) {
        const kernel = weightGradientKernel(a_prev.length, delta.length);
        const result = kernel(a_prev, delta);

        return Array.from(result).map(row => Array.from(row)); 
    } else {
        // Fallback: manually compute outer product
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
}


/**
 * Scale gradients (1D or 2D)
 */
function scaleGradients(onGPU, grad, scalar) {
    if (onGPU) {
        const kernel = scaleKernel(grad.length);
        return Array.from(kernel(grad, scalar));
    } else {
        if (Array.isArray(grad[0])) {
            // 2D array
            return grad.map(row => row.map(v => v / scalar));
        } else {
            // 1D array
            return grad.map(v => v / scalar);
        }
    }
}

module.exports = {
    computeWeightGradients,
    scaleGradients,
    gpuInstance: gpu
};

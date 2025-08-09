/**
 * Matrix operations for dot products in Neurex
 */
const {gpu} = require('../gpu-init');

/**
 * Forward pass kernel: z = Wᵀ · input + b
 */
const forwardKernel = (inputSize, outputSize) => {
    return gpu.createKernel(function(inputs, weights, biases) {
        let sum = biases[this.thread.x];
        for (let i = 0; i < this.constants.inputSize; i++) {
            sum += inputs[i] * weights[i][this.thread.x];
        }
        return sum;
    }, {
        output: [outputSize],
        constants: { inputSize }
    });
};

/**
 * Backprop delta kernel: δᵢ = Σ δₖ * wᵢₖ
 */
const backpropKernel = (nextSize, currentSize) => {
    return gpu.createKernel(function(weightsNext, deltaNext) {
        let sum = 0;
        for (let i = 0; i < this.constants.nextSize; i++) {
            sum += weightsNext[this.thread.x][i] * deltaNext[i];
        }
        return sum;
    }, {
        output: [currentSize],
        constants: { nextSize }
    });
};

/**
 * Compute forward dot product for one layer (GPU or CPU)
 */
const computeForward = (onGPU, input, weights, biases) => {
    if (onGPU) {
        const kernel = forwardKernel(input.length, biases.length);
        return Array.from(kernel(input, weights, biases));
    } else {
        let output = [];
        for (let neuron = 0; neuron < biases.length; neuron++) {
            let sum = biases[neuron];
            for (let i = 0; i < input.length; i++) {
                sum += input[i] * weights[i][neuron];
            }
            output.push(sum);
        }
        return output;
    }
};

/**
 * Compute backpropagated deltas (GPU or CPU)
 */
const computeBackprop = (onGPU, weightsNext, deltaNext) => {
    if (onGPU) {
        const kernel = backpropKernel(deltaNext.length, weightsNext.length);
        return Array.from(kernel(weightsNext, deltaNext));
    } else {
        let output = [];
        for (let i = 0; i < weightsNext.length; i++) {
            let sum = 0;
            for (let j = 0; j < deltaNext.length; j++) {
                sum += weightsNext[i][j] * deltaNext[j];
            }
            output.push(sum);
        }
        return output;
    }
};

module.exports = {
    computeForward,
    computeBackprop,
    gpuInstance: gpu
};

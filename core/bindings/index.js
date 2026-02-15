let path = require('path');


let addon;

try {
    addon = require(path.join(__dirname, 'prebuilds', `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
}
catch (error) {
    console.error(error);
}


/**
 * @function MatMul
 * @param {Array<Number>} inputs - 1D array of input features
 * @param {Array<Array<Number>>} weights - 2D array of weights
 * @param {Array<Number>} biases - 1D array of biases
 * @returns 1D array of output
 */
const MatMul = (inputs, weights, biases) => addon.MatMul(inputs, weights, biases);

/**
 * @function DeltaMatMul
 * @param {Array<Array<Number>>} weights - 2D array of weights
 * @param {Array<Number>} deltas - 1D array of output deltas from the previous layer
 * @returns 1D array of output deltas of the current layer to be use to the next layer during backpropagation
 */
const DeltaMatMul = (weights, deltas) => addon.DeltaMatMul(weights, deltas);

/**
 * @function relu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu)
 */
const relu = (input) => addon.Relu(input);

/**
 * @function sigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid)
 */
const sigmoid = (input) => addon.Sigmoid(input);

/**
 * @function tanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh)
 */
const tanh = (input) => addon.Tanh(input);

/**
 * @function softmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax)
 */
const softmax = (input) => addon.Softmax(input);

/**
 * @function linear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear)
 */
const linear = (input) => addon.Linear(input);

/**
 * @function drelu
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using ReLu Derivative)
 */
const drelu = (input) => addon.DReLu(input);

/**
 * @function dsigmoid
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Sigmoid Derivative)
 */
const dsigmoid = (input) => addon.DSigmoid(input);

/**
 * @function dtanh
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Tanh Derivative)
 */
const dtanh = (input) => addon.DTanh(input);

/**
 * @function dsoftmax
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Softmax Derivative)
 */
const dsoftmax = (input) => addon.DSoftmax(input);

/**
 * @function dlinear
 * @param {Array<Number>} input - 1D array of features 
 * @returns - 1D array of activated features (Using Linear Derivative)
 */
const dlinear = (input) => addon.DLinear(input);

/**
 * 
 * @param {Number} filters 
 * @param {Number} strides 
 * @param {Array<Array<Array<Array<Number>>>>>} input 
 * @param {Array<Array<Array<Number>>>} kernels 
 * @param {Array<Number>} biases 
 * @param {Number} OutputHeight
 * @param {Number} OutputWidth
 * @returns array of 4D feature maps
 */

// let idx = 1;
const Convolve = (strides = 1,input, kernels, biases, OutputHeight, OutputWidth) => addon.Convolve(strides, input, kernels, biases, OutputHeight, OutputWidth);
/**
 * 
 * @param {Array<Array<Array<Array<Number>>>>} featureMaps 
 * @returns 1 stack of feature map with increasing depth
 */
const StackFeatureMaps = (featureMaps) => addon.StackFeatureMaps(featureMaps);

/// ======================================= ///

// rotateKernels and ConvolveDelta() needs a native code with same functionality
// Rotates each kernel by 180 degrees
function rotateKernels(kernels) {
    const F  = kernels.length;
    const KH = kernels[0].length;
    const KW = kernels[0][0].length;
    const D  = kernels[0][0][0].length;

    const rotated = Array.from({ length: F }, () =>
        Array.from({ length: KH }, () =>
            Array.from({ length: KW }, () =>
                Array(D).fill(0)
            )
        )
    );

    for (let f = 0; f < F; f++) {
        for (let kh = 0; kh < KH; kh++) {
            for (let kw = 0; kw < KW; kw++) {
                for (let d = 0; d < D; d++) {
                    rotated[f][KH - 1 - kh][KW - 1 - kw][d] =
                        kernels[f][kh][kw][d];
                }
            }
        }
    }
    return rotated;
}

/**
 * Performs backpropagation convolution to find the delta of the previous layer.
 * @param {Array} padded_dilated_delta - The delta from the next layer, already dilated and padded.
 * @param {Array} kernels - The kernels/weights of the current layer [Filters][Channels][Height][Width]
 * @param {Number} inputH - The height of the input from the forward pass we are trying to reach.
 * @param {Number} inputW - The width of the input from the forward pass we are trying to reach.
 * @returns {Array} 3D Tensor [inputH][inputW][Channels]
 */
function ConvolveDelta(padded_dilated_delta, kernels, inputH, inputW, layerIdx) {
    const F  = kernels.length;
    const KH = kernels[0].length;
    const KW = kernels[0][0].length;
    const D  = kernels[0][0][0].length;

    if (padded_dilated_delta.flat(Infinity).some(isNaN)) console.log(`layer ${layerIdx} has NaNs on it's padded_dilated_delta`)


    const rotated = rotateKernels(kernels);

    let delta = Array.from({ length: inputH }, () =>
        Array.from({ length: inputW }, () =>
            Array(D).fill(0)
        )
    );

    // For each channel in input
    for (let d = 0; d < D; d++) {
        for (let h = 0; h < inputH; h++) {
            for (let w = 0; w < inputW; w++) {
                let sum = 0;
                // For each filter
                for (let f = 0; f < F; f++) {
                    const kernel = rotated[f];
                    // Convolve kernel with padded_dilated_delta
                    for (let kh = 0; kh < KH; kh++) {
                        for (let kw = 0; kw < KW; kw++) {
                            // Calculate the position in padded_dilated_delta
                            const ph = h + kh;
                            const pw = w + kw;
                            let a = 0;
                            if (
                                ph >= 0 && ph < padded_dilated_delta.length &&
                                pw >= 0 && pw < padded_dilated_delta[0].length &&
                                f >= 0 && f < padded_dilated_delta[0][0].length
                            ) {
                                a = padded_dilated_delta[ph][pw][f];
                                if (a === undefined || isNaN(a)) {
                                    a = 0;
                                };
                            }
                            const b = kernel[kh][kw][d];
                            if (b === undefined || isNaN(b)) {
                                console.warn(`NaN or undefined detected in kernel: b=${b}, kh=${kh}, kw=${kw}, d=${d}`);
                                continue;
                            }
                            sum += a * b;
                        }
                    }
                }
                delta[h][w][d] = sum;
            }
        }
    }

    return delta;
}

/**
 * 
 * @param {*} activated_outputs 
 * @param {*} delta 
 * @param {*} layer_name 
 * @param {*} weightGrads
 * @param {*} layer_data
 * @returns 
 */
const computeWeightGradients = (activated_outputs, delta, layer_name, weightGrads, layer_data, allDeltas, layer_index) => {
    // if (activated_outputs.flat(Infinity).some(isNaN)) throw new Error(`Error occured when computing weight gradients on layer ${layer_data.layer_name} ${layer_index+1}. Reason: "activated_outpus" has NaNs. If this error occured, please report this error`);
    // if (delta.flat(Infinity).some(isNaN)) throw new Error(`Error occured when computing weight gradients on layer ${layer_data.layer_name} ${layer_index+1}. Reason: "delta" has NaNs. If this error occured, please report this error`); 
    // if (weightGrads.flat(Infinity).some(isNaN)) throw new Error(`Error occured when computing weight gradients on layer ${layer_data.layer_name} ${layer_index+1}. Reason: "weightGrads" has NaNs. If this error occured, please report this error`); 


    if (layer_name === "convolutional2D") {

        if (layer_index == 0) return weightGrads;
    
        return ComputeGradientForKernels(activated_outputs, delta, weightGrads);
    }
    else if (layer_name === "connected_layer") {
        let input = activated_outputs;
        if (Array.isArray(activated_outputs[0] || Array.isArray(activated_outputs[0][0]))) {
            input = activated_outputs.flat(Infinity);
        }

        return addon.ComputeGradientForDenseWeights(input, delta, weightGrads);

    }

    else {
        return weightGrads;
    }
}


/**
 * 
 * @param {*} weightGrads 
 * @param {*} actualBatchSize 
 * @param {*} layer_name 
 * @returns 
 */
const scaleGradientsForWeights = (weightGrads, actualBatchSize, layer_name) => {
    
    if (layer_name === "connected_layer") {

        return scaleGradientWeightsForConnectedLayer(weightGrads, actualBatchSize);

    }
    
    if (layer_name === "convolutional2D") {
        return scaleGradientWeightsForConv(weightGrads, actualBatchSize);
    }

    else {
        return weightGrads;
    }
}

/**
 * 
 * @param {*} biasGrads 
 * @param {*} delta 
 * @param {*} layer_name 
 * @returns 
 */
const computeBiasGradients = (biasGrads, delta, layer_name) => {

    if (layer_name === "connected_layer") {
        return ComputeGradientsForDenseBiases(biasGrads, delta);
    }
    else if (layer_name === "convolutional2D") {
        return ComputeGradientsForConvBiases(biasGrads, delta);
    }
}

/**
 * 
 * @param {*} grad 
 * @param {*} scalar 
 * @returns 
 */
const scaleGradientsForBiases = (grad, scalar) => {

    return grad.map(v => v / scalar);
}

// need to write a native binding for this but for testing, we implement it for now in plain JS
const ComputeGradientForKernels = (activated_outputs, delta, weightGrads) => {
    const filters = weightGrads.length;
    const kernel_height = weightGrads[0].length;
    const kernel_width = weightGrads[0][0].length;
    const channels = weightGrads[0][0][0].length;

    // Output spatial dimensions
    const out_height = delta.length;
    const out_width = delta[0].length;

    // Zero-initialized gradient accumulator
    let output_grad = Array.from({ length: filters }, () =>
        Array.from({ length: kernel_height }, () =>
            Array.from({ length: kernel_width }, () =>
                Array(channels).fill(0)
            )
        )
    );

    // Accumulate gradients with robust checks
    for (let f = 0; f < filters; f++) {
        for (let kh = 0; kh < kernel_height; kh++) {
            for (let kw = 0; kw < kernel_width; kw++) {
                for (let c = 0; c < channels; c++) {
                    let grad = 0;
                    for (let i = 0; i < out_height; i++) {
                        for (let j = 0; j < out_width; j++) {
                            const input_i = i + kh;
                            const input_j = j + kw;
                            let a = 0, b = 0;
                            if (
                                input_i >= 0 && input_i < activated_outputs.length &&
                                input_j >= 0 && input_j < activated_outputs[0].length
                            ) {
                                const valA = activated_outputs[input_i][input_j] && activated_outputs[input_i][input_j][c];
                                a = (typeof valA === 'number' && !isNaN(valA)) ? valA : 0;
                            }
                            if (
                                delta[i] && delta[i][j] && typeof delta[i][j][f] === 'number' && !isNaN(delta[i][j][f])
                            ) {
                                b = delta[i][j][f];
                            }
                            grad += a * b;
                        }
                    }
                    output_grad[f][kh][kw][c] = grad;
                }
            }
        }
    }


    if (output_grad.flat(Infinity).some(isNaN)) throw new Error('Error on gradient accumulator. Contains NaNs')
    return output_grad;
};

// need to write a native binding for this but for testing, we implement it for now in plain JS
const ComputeGradientsForDenseBiases = (biasGrads, delta) => {

    let output_grad = biasGrads;
    for (let j = 0; j < biasGrads.length; j++) {
        output_grad[j] += delta[j];
    }

    return output_grad;
}

// need to write a native binding for this but for testing, we implement it for now in plain JS
const ComputeGradientsForConvBiases = (biasGrads, delta) => {
    const output_grad = biasGrads.slice(); // shallow copy to avoid mutation
    for (let f = 0; f < output_grad.length; f++) {
        for (let h = 0; h < delta.length; h++) {
            for (let w = 0; w < delta[0].length; w++) {
                const v = delta[h][w][f];
                if (typeof v === 'number' && !isNaN(v)) {
                    output_grad[f] += v;
                }
            }
        }
    }
    return output_grad;
}

// need to write a native binding for this but for testing, we implement it for now in plain JS
const scaleGradientWeightsForConnectedLayer = (weightGrads, actualBatchSize) => {

    const scaled_output = weightGrads;

    for (let i = 0; i < weightGrads.length; i++) {
       
        for (let j = 0; j < weightGrads[i].length; j++) {
            scaled_output[i][j] /= actualBatchSize;
        }
    }   

    return scaled_output;

}

// need to write a native binding for this but for testing, we implement it for now in plain JS
const scaleGradientWeightsForConv = (weightGrads, actualBatchSize) => {

    const scaled_kernels = weightGrads;

    for (let f = 0; f < weightGrads.length; f++) {
        for (let kh = 0; kh < weightGrads[0].length; kh++) {
            for (let kw = 0; kw <  weightGrads[0][0].length; kw++) {
                for (let d = 0; d < weightGrads[0][0][0].length; d++) {
                    scaled_kernels[f][kh][kw][d] /= actualBatchSize;
                }
            }
        }
    }

    return scaled_kernels;
}




module.exports = {
    MatMul,
    DeltaMatMul,
    relu,
    sigmoid,
    tanh,
    softmax,
    linear,
    Convolve,
    StackFeatureMaps,
    ConvolveDelta,
    computeWeightGradients,
    computeBiasGradients,
    scaleGradientsForWeights,
    scaleGradientsForBiases,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
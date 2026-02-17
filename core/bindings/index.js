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

const Convolve = (strides = 1,input, kernels, biases, OutputHeight, OutputWidth) => addon.Convolve(strides, input, kernels, biases, OutputHeight, OutputWidth);

/**
 * 
 * @param {Array<Array<Array<Array<Number>>>>} featureMaps 
 * @returns 1 stack of feature map with increasing depth
 */
const StackFeatureMaps = (featureMaps) => addon.StackFeatureMaps(featureMaps);

/**
 * Performs backpropagation convolution to find the delta of the previous layer.
 * @param {Array} padded_dilated_delta - The delta from the next layer, already dilated and padded.
 * @param {Array} kernels - The kernels/weights of the current layer [Filters][Channels][Height][Width]
 * @param {Number} inputH - The height of the input from the forward pass we are trying to reach.
 * @param {Number} inputW - The width of the input from the forward pass we are trying to reach.
 * @returns {Array} 3D Tensor [inputH][inputW][Channels]
 */
const ConvolveDelta = (padded_dilated_delta, kernels, inputH, inputW) => addon.ConvolveDelta(padded_dilated_delta, kernels, inputH, inputW);


/**
 * 
 * @param {Array<Number>} params - flattened array of parameters 
 * @param {Array<Number>} grads - flattened array of grads 
 * @param {Number} learning_rate - learning rate value
 * @returns 
 */
const ApplySGD = (params, grads, learning_rate ) => addon.SGD(params, grads, learning_rate);

/**
 * 
 * @param {Array<Number>} params - flattened array of parameters 
 * @param {Array<Number>} grads - flattened array of grads  
 * @param {Array<Number>} m - first momentum of average gradients vector
 * @param {Array<Number>} v - second momentum of squared average gradients vector
 * @param {Number} t - Time step counter 
 * @param {Number} learning_rate - learning rate value 
 * @param {Number} beta1 - beta1 value
 * @param {Number} beta2 - beta2 value
 * @param {Number} epsilon - epsilon constant value
 * @returns 
 */
const ApplyAdam = (params, grads, m, v, t, learning_rate, beta1, beta2, epsilon) => addon.Adam(params, grads, m, v, t, learning_rate, beta1, beta2, epsilon);

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

    if (layer_name === "convolutional2D") {

        if (layer_index == 0) return weightGrads;
    
        return addon.ComputeGradientForKernels(activated_outputs, delta, weightGrads);
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
    ApplySGD,
    ApplyAdam,
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
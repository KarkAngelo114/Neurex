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
function ConvolveDelta(padded_dilated_delta, kernels, inputH, inputW) {
    const F  = kernels.length;
    const KH = kernels[0].length;
    const KW = kernels[0][0].length;
    const D  = kernels[0][0][0].length;


    const rotated = rotateKernels(kernels);

    //console.log('Inside convolve delta');
    //console.log('Padded dilated delta shape:', padded_dilated_delta.length, padded_dilated_delta[0].length, padded_dilated_delta[0][0].length);
    //console.log('Kernel shape:', rotated.length, rotated[0].length, rotated[0][0].length, rotated[0][0][0].length);
    //console.log('Expected output height and width:', inputH, inputW);    
    // Initialize the output delta: [Height][Width][Depth/Channels]
    const deltaX = Array.from({ length: inputH }, () =>
        Array.from({ length: inputW }, () =>
            Array(D).fill(0.0)
        )
    );

    // Standard valid convolution:
    // We slide over the padded_dilated_delta to produce an output of size [inputH][inputW]
    for (let i = 0; i < inputH; i++) {
        for (let j = 0; j < inputW; j++) {
            for (let d = 0; d < D; d++) { // For each input channel
                let sum = 0.0;
                for (let f = 0; f < F; f++) { // Sum across all filters
                    for (let kh = 0; kh < KH; kh++) {
                        for (let kw = 0; kw < KW; kw++) {
                            sum += padded_dilated_delta[i + kh][j + kw][f] * rotated[f][kh][kw][d];
                        }
                    }
                }
                deltaX[i][j][d] = sum;
            }
        }
    }
    return deltaX;
}

// const ConvolveDelta = (delta_feature_maps, kernels, strides, top, bottom, left, right, inputH, inputW) => addon.ConvolveDelta(delta_feature_maps, kernels, strides, top, bottom, left, right, inputH, inputW);

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
    
        // const output = addon.ComputeGradientForKernels(activated_outputs, delta, weightGrads, 1);
        // return output

        if (layer_index == 0) return weightGrads;
    
        return output = ComputeGradientForKernels(activated_outputs, delta, weightGrads);
        // return weightGrads
    }
    else if (layer_name === "connected_layer") {
        let input = activated_outputs;
        if (Array.isArray(activated_outputs[0] || Array.isArray(activated_outputs[0][0]))) {
            input = activated_outputs.flat(Infinity);
        }
        //console.log(allDeltas[layer_index-1])
        const output = addon.ComputeGradientForDenseWeights(input, delta, weightGrads);
        return output
    }

    else {
        return weightGrads;
    }
}

// need to write a native binding for this
const ComputeGradientForKernels = (activated_outputs, delta, weightGrads, stride = 1) => {
    const filters = weightGrads.length;
    const kernel_height = weightGrads[0].length;
    const kernel_width = weightGrads[0][0].length;
    const channels = weightGrads[0][0][0].length;

    // Create a zero-initialized gradient accumulator
    let output_grad = weightGrads.map(f =>
        f.map(kh =>
            kh.map(kw =>
                kw.slice()
            )
        )
    );

    for (let f = 0; f < filters; f++) {
        for (let c = 0; c < channels; c++) {
            for (let kh = 0; kh < kernel_height; kh++) {
                for (let kw = 0; kw < kernel_width; kw++) {
                    let grad = 0;
                    for (let i = 0; i <= activated_outputs.length - kernel_height; i += stride) {
                        for (let j = 0; j <= activated_outputs[0].length - kernel_width; j += stride) {
                            grad += activated_outputs[i + kh][j + kw][c] * delta[i][j][f];
                        }
                    }
                    output_grad[f][kh][kw][c] += grad;
                }
            }
        }
    }
    return output_grad;
};


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
    derivatives: {
        relu: drelu,
        sigmoid: dsigmoid,
        tanh: dtanh,
        softmax: dsoftmax,
        linear: dlinear
    },
}
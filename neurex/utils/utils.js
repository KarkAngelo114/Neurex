const getShape = (arr) => {
    const shape = [];
    let curr = arr;
    while (Array.isArray(curr)) {
        shape.push(curr.length);
        curr = curr[0];
    }

    return shape;
};

const flattenAll = (params) => {
    return params.flat(Infinity);
}

const calculateTensorShape = (inputHeight, inputWidth, kernelHeight, kernelWidth, depth, stride, padding) => {
    // console.log(inputHeight, inputWidth, kernelHeight, kernelWidth, depth, stride, padding);
    let oH, oW;
    if (padding === "same") {
        oH = Math.ceil(inputHeight / stride);
        oW = Math.ceil(inputWidth / stride);
    } else {
        oH = Math.floor((inputHeight - kernelHeight) / stride + 1);
        oW = Math.floor((inputWidth - kernelWidth) / stride + 1);
    }

    return {
        OutputHeight: oH,
        OutputWidth: oW,
        CalculatedTensorShape: oH * oW * depth
    };
};

/**
 * Pads a 3D Tensor [H][W][C]
 * @param {Array} input - The 3D array
 * @param {number} padTop - Padding amounts
 * @param {number} padBottom - Padding amounts
 * @param {number} padLeft - Padding amounts
 * @param {number} padRight - Padding amounts
 * @returns {Array} The padded 3D array
 */
const applyPadding = (input, padTop, padBottom, padLeft, padRight) => {
    const inputH = input.length;
    const inputW = input[0].length;
    const channels = input[0][0].length;

    const newH = inputH + padTop + padBottom;
    const newW = inputW + padLeft + padRight;

    // Create a new 3D tensor filled with zeros
    // Using Array.from is cleaner for initializing nested arrays
    const output = Array.from({ length: newH }, () =>
        Array.from({ length: newW }, () => 
            new Array(channels).fill(0)
        )
    );

    // Fill the inner part with the original input data
    for (let i = 0; i < inputH; i++) {
        for (let j = 0; j < inputW; j++) {
            output[i + padTop][j + padLeft] = input[i][j];
        }
    }

    return output;
}

/**
 * 
 * @param {Number} inputH - height of the input
 * @param {Number} inputW - width of the input 
 * @param {Number} kernelH - height of the kernel
 * @param {Number} kernelW - width of the kernel
 * @param {Number} stride - stride of the
 * @param {String} padding - "same" or "valid"
 * @returns 
 */
const getPaddingSizes = (inputH, inputW, kernelH, kernelW, stride, padding) => {
    if (padding === "valid") {
        return { top: 0, bottom: 0, left: 0, right: 0 };
    }

    // Standard formula for total padding needed
    const outputH = Math.ceil(inputH / stride);
    const outputW = Math.ceil(inputW / stride);

    const padH = Math.max(0, (outputH - 1) * stride + kernelH - inputH);
    const padW = Math.max(0, (outputW - 1) * stride + kernelW - inputW);

    // Distribute padding to sides (asymmetric if necessary)
    return {
        top: Math.floor(padH / 2),
        bottom: padH - Math.floor(padH / 2),
        left: Math.floor(padW / 2),
        right: padW - Math.floor(padW / 2)
    };
};

/**
 * 
 * @param {Array<Array<Array<Number>>>} input - delta tensor map as input
 * @param {Number} stride - strides given for how much the kernel moves across the input tensor 
 * @returns Dilated output
 */
const DilateInput = (input, stride) => {
    if (stride <= 1) return input;

    const inputH = input.length;
    const inputW = input[0].length;
    const inputD = input[0][0].length;

    

    const newH = inputH + (inputH - 1) * (stride - 1);
    const newW = inputW + (inputW - 1) * (stride  - 1);

    const output = Array.from({length: newH},() => Array.from({length: newW}, () => Array.from({length: inputD}).fill(0)));

    for (let i = 0; i < inputH; i++) {
        for (let j = 0; j < inputW; j++) {
            output[i * stride][j * stride] = input[i][j];
        }
    }

    return output;

}

module.exports = {
    getShape,
    flattenAll,
    calculateTensorShape,
    applyPadding,
    getPaddingSizes,
    DilateInput
}
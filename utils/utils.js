

const XavierInitialization = (inputSize, outputSize) => {
    return Math.sqrt(2 / (inputSize + outputSize));
}

const calculateTensorShape = (inputHeight, inputWidth, kernelHeight, kernelWidth, outputdepth, stride, padding) => {
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
        CalculatedTensorShape: oH * oW * outputdepth
    };
};

/**
 * 
 * @param {Number} inputHeight 
 * @param {Number} inputWidth 
 * @param {Number} kernelHeight 
 * @param {Number} kernelWidth 
 * @param {Number} outputDepth 
 * @param {Number} stride 
 * @param {String} padding 
 * @returns {{OutputHeight: Number, OutputWidth: Number, CalculatedTensorShape: Number}}
 */
const calculateTransposedTensorShape = (inputHeight, inputWidth, kernelHeight, kernelWidth, outputDepth, stride = 1,  padding = "same") => {
    let oH, oW;

    if (padding === "same") {
        // "SAME" padding aims to scale spatial dimensions directly by the stride
        oH = inputHeight * stride;
        oW = inputWidth * stride;
    } else {
        // "VALID" padding (no extra padding added to the output boundary)
        oH = (inputHeight - 1) * stride + kernelHeight;
        oW = (inputWidth - 1) * stride + kernelWidth;
    }

    return {
        OutputHeight: oH,
        OutputWidth: oW,
        CalculatedTensorShape: oH * oW * outputDepth
    };
};

/**
 * 
 * @param {Number} inputH 
 * @param {Number} inputW 
 * @param {Number} kernelH 
 * @param {Number} kernelW 
 * @param {Number} stride 
 * @param {String} padding 
 * @returns {{top: Number, left: Number, dilatedH: number, dilatedW: Number}}
 */
const getTransposedPaddingSizes = (inputH, inputW, kernelH, kernelW, stride, padding) => {
    const dilatedH = (inputH - 1) * stride + 1;
    const dilatedW = (inputW - 1) * stride + 1;

    const targetH = padding === "valid" ? (inputH - 1) * stride + kernelH : inputH * stride;
    const targetW = padding === "valid" ? (inputW - 1) * stride + kernelW : inputW * stride;

    const totalPadH = targetH - dilatedH + kernelH - 1;
    const totalPadW = targetW - dilatedW + kernelW - 1;

    return {
        top: Math.floor(totalPadH / 2), bottom: totalPadH - Math.floor(totalPadH / 2),
        left: Math.floor(totalPadW / 2), right: totalPadW - Math.floor(totalPadW / 2),
        dilatedH: dilatedH, 
        dilatedW: dilatedW
    };
};

/**
 * 
 * @param {Number} inputH - height of the input
 * @param {Number} inputW - width of the input 
 * @param {Number} kernelH - height of the kernel
 * @param {Number} kernelW - width of the kernel
 * @param {Number} stride - stride value
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
}

const ifOneHotEndcoded = (Y_train) => {
        /**
        Checks if all rows in Y_train are one-hot encoded.
        Each row must:
        - Contain only 0s and 1s
        - Have exactly one "1"
        */
        for (let i = 0; i < Y_train.length; i++) {
            const row = Y_train[i];
            if (!Array.isArray(row)) return false;

            let onesCount = 0;
            for (let j = 0; j < row.length; j++) {
                if (row[j] !== 0 && row[j] !== 1) return false;
                if (row[j] === 1) onesCount++;
            }

            if (onesCount !== 1) return false;
        }
        return true;
    }

const getTotalMB = (array) => {
    let sum = 0;
    for (let i = 0; i < array.length; i++) {
        sum += array[i].byteLength / (1024 * 1024);
    }
    return sum;
}

const formatDuration = (totalSeconds) => {
    const d = Math.floor(totalSeconds / (3600 * 24));
    const h = Math.floor((totalSeconds % (3600 * 24)) / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60; 

    const parts = [];
    if (d > 0) parts.push(`${d}d`);
    if (h > 0) parts.push(`${h}h`);
    if (m > 0) parts.push(`${m}m`);
    
    // Use .toFixed(1) for one decimal place (e.g., 0.2s)
    if (s > 0 || parts.length === 0) {
        parts.push(`${s.toFixed(3)}s`);
    }

    return parts.join(' ');
}

/**
 * 
 * @param {Array<Float32Array>} chunks an array collection of float32 array 
 * @returns { Float32Array }
 */
const concatenateFloat32Array = (chunks) => {
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Float32Array(totalLength);
    
    // 2. Copy each chunk into the target position
    let offset = 0;
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    
    return result;
}

/**
 * @function unpackQKV allows you to unpack parameters of weights and biases of Q, K and V as well as the weights and biases zeroed accumulator
 * @param {Float32Array} weights QKV weights concatenated to one contagious array. (Nullable)
 * @param {Float32Array} biases QKV biases concatenated to one contagious array. (Nullable)
 * @param {Float32Array} weightGrads QKV weightsGrads concatenated to one contagious array. (Nullable)
 * @param {Float32Array} biasesGrads QKV biasesGrads concatenated to one contagious array. (Nullable)
 * @param {Number} embeddingDim embedding dimension value
 * @returns {{ 
 *   Q_weights: Float32Array, 
 *   K_weights: Float32Array, 
 *   V_weights: Float32Array, 
 *   Q_bias: Floa32Array, 
 *   K_bias: Float32Array, 
 *   V_bias: Float32Array,
 *   Q_weightGrads: Float32Array,
 *   K_weightGrads: Float32Array,
 *   V_weightGrads: Float32Array,
 *   Q_biasGrads: Float32Array,
 *   K_biasGrads: Float32Array,
 *   V_biasGrads: Float32Array,
 * }}
 *
 *
 *
 *
 */
const unpackQKV = (weights, biases, weightGrads, biasGrads, embeddingDim) => {
    const matrixSize = embeddingDim * embeddingDim;
    let Qw; // Q_weights
    let Kw; // K_weights
    let Vw; // V_weights
    let Qb; // Q_bias
    let Kb; // K_bias
    let Vb; // V_bias
    let QwG; // Q_weightGrads
    let KwG; // K_weightGrads
    let VwG; // V_weightGrads
    let QbG; // Q_biasGrads
    let KbG; // K_biasGrads
    let VbG; // V_biasGrads

    if (weights) {
        Qw = weights.subarray(0, matrixSize);
        Kw = weights.subarray(matrixSize, matrixSize * 2);
        Vw = weights.subarray(matrixSize * 2, matrixSize * 3);
    }

    if (biases) {
        Qb = biases.subarray(0, embeddingDim);
        Kb = biases.subarray(embeddingDim, embeddingDim * 2);
        Vb = biases.subarray(embeddingDim * 2, embeddingDim * 3);
    }

    if (weightGrads) {
        QwG = weightGrads.subarray(0, matrixSize);
        KwG = weightGrads.subarray(matrixSize, matrixSize * 2);
        VwG = weightGrads.subarray(matrixSize * 2, matrixSize * 3);
    }

    if (biasGrads) {
        QbG = biasGrads.subarray(0, embeddingDim);
        KbG = biasGrads.subarray(embeddingDim, embeddingDim * 2);
        VbG = biasGrads.subarray(embeddingDim * 2, embeddingDim * 3);
    }

    return { 
        Q_weights: Qw, 
        K_weights: Kw, 
        V_weights: Vw, 
        Q_bias: Qb, 
        K_bias: Kb, 
        V_bias: Vb,
        Q_weightGrads: QwG,
        K_weightGrads: KwG,
        V_weightGrads: VwG,
        Q_biasGrads: QbG,
        K_biasGrads: KbG,
        V_biasGrads: VbG
    };
};

/**
 * Transposes a flattened 1D Float32Array representing a 2D matrix [rows x cols]
 * @param {Float32Array} matrix - Flattened matrix
 * @param {Number} rows - Number of rows
 * @param {Number} cols - Number of columns
 * @returns {Float32Array} Transposed flattened matrix [cols x rows]
 */
function transpose2D(matrix, rows, cols) {
    const output = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            output[c * rows + r] = matrix[r * cols + c];
        }
    }
    return output;
}

module.exports = {
    calculateTensorShape,
    calculateTransposedTensorShape,
    getPaddingSizes,
    XavierInitialization,
    ifOneHotEndcoded,
    getTotalMB,
    formatDuration,
    concatenateFloat32Array,
    getTransposedPaddingSizes,
    unpackQKV,
    transpose2D
}
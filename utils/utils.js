

const XavierInitialization = (inputSize, outputSize) => {
    return Math.sqrt(2 / (inputSize + outputSize));
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

module.exports = {
    calculateTensorShape,
    getPaddingSizes,
    XavierInitialization,
    ifOneHotEndcoded,
    getTotalMB,
    formatDuration
}
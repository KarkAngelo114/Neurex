

const convolve = (onGPU, filters, strides, input, kernels, biases, padding = "valid") => {
    if (onGPU) {
        // TODO
    }
    else {
        console.log('biases',biases)
        let input_depth = input[0][0].length;
        let kernelHeight = kernels[0].length;
        let kernelWidth = kernels[0][0].length;

        let pad = 0;
        if (padding === "same") {
            pad = Math.floor(kernelHeight / 2);
        }

        let paddedInput = pad > 0 ? applyPadding(input, pad) : input;

        const padded_input_height = paddedInput.length;
        const padded_input_width = paddedInput[0].length;

        const outputHeight = Math.floor((padded_input_height - kernelHeight) / strides + 1);
        const outputWidth = Math.floor((padded_input_width - kernelWidth) / strides + 1);

        const feature_maps = [];

        for (let filterIndex = 0; filterIndex < kernels.length; filterIndex++) {
            const kernel = kernels[filterIndex];
            const bias = biases[filterIndex];

            let output = [];
            for (let height = 0; height < outputHeight; height++) {
                let row = [];
                for (let width = 0; width < outputWidth; width++) {
                    let sum = 0;
                    for (let kernel_Height = 0; kernel_Height < kernelHeight; kernel_Height++) {
                        for (let kernel_width = 0; kernel_width < kernelWidth; kernel_width++) {
                            for (let depth = 0; depth < input_depth; depth++) {
                                const inputVal = paddedInput[height * strides + kernel_Height][width * strides + kernel_width][depth];
                                const kernel_val = kernel[kernel_Height][kernel_width][depth];
                                sum += inputVal * kernel_val;
                            }
                        }
                    }
                    sum += bias;
                    row.push([sum]);
                }
                output.push(row)
            }
            feature_maps.push(output);
        }

        return feature_maps;
    }
};

const convolveDelta = (delta_map, kernels) => {
    // delta_map: H × W × F
    // kernels:   F × KH × KW × D

    const H = delta_map.length;
    const W = delta_map[0].length;
    const F = delta_map[0][0].length;

    const KH = kernels[0].length;
    const KW = kernels[0][0].length;
    const D  = kernels[0][0][0].length;

    // Rotate kernels (180°)
    const rotated = rotateKernel(kernels);

    // Padding = kernel_size - 1
    const padH = KH - 1;
    const padW = KW - 1;

    // Initialize δX
    const deltaX = Array(H)
        .fill(0)
        .map(() =>
            Array(W)
                .fill(0)
                .map(() => Array(D).fill(0))
        );

    // For each input depth
    for (let d = 0; d < D; d++) {

        // Accumulate over all filters
        for (let f = 0; f < F; f++) {

            // Extract δY[:,:,f]
            const delta2D = delta_map.map(row => row.map(cell => cell[f]));

            // Pad δY
            const paddedDelta = pad2D(delta2D, padH, padW);

            // Kernel slice: rotated[f][:,:,d]
            const kernel2D = rotated[f].map(row =>
                row.map(cell => cell[d])
            );

            // Convolution
            const conv = conv2D(paddedDelta, kernel2D);

            // Accumulate into δX[:,:,d]
            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    deltaX[i][j][d] += conv[i][j];
                }
            }
        }
    }

    return deltaX;
};

function conv2D(input, kernel) {
    const H = input.length;
    const W = input[0].length;
    const KH = kernel.length;
    const KW = kernel[0].length;

    const outH = H - KH + 1;
    const outW = W - KW + 1;

    const output = Array(outH)
        .fill(0)
        .map(() => Array(outW).fill(0));

    for (let i = 0; i < outH; i++) {
        for (let j = 0; j < outW; j++) {
            let sum = 0;
            for (let ki = 0; ki < KH; ki++) {
                for (let kj = 0; kj < KW; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}


function pad2D(mat, padH, padW) {
    const H = mat.length;
    const W = mat[0].length;

    const padded = Array(H + 2 * padH)
        .fill(0)
        .map(() => Array(W + 2 * padW).fill(0));

    for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
            padded[i + padH][j + padW] = mat[i][j];
        }
    }
    return padded;
}

    

const rotateKernel = (kernel) => {

    let rotatedKernels = [];

    for (let f = 0; f < kernel.length; f++) {
        let K = kernel[f]; // we need first to access the i-th kernel from the array of kernels for this layer.

        // and rotate the kernel by reversing the rows, iterate to every rows (using map()) and reverse them to make the kernel rotate 180 degree
        const rotated_kernel = K.slice().reverse().map(innerElements => innerElements.slice().reverse())
        rotatedKernels.push(rotated_kernel);
    }
    return rotatedKernels;
}

const applyPadding = (input, pad) => {
    const inputHeight = input.length;
    const inputWidth = input[0].length;
    const channels = input[0][0].length;

    const output = [];

    for (let i = 0; i < inputHeight + 2 * pad; i++) {
        const row = [];
        for (let j = 0; j < inputWidth + 2 * pad; j++) {
            if (i < pad || i >= inputHeight + pad || j < pad || j >= inputWidth + pad) {
                // Push zero array for each channel
                row.push(Array(channels).fill(0));
            } else {
                row.push(input[i - pad][j - pad]);
            }
        }
        output.push(row);
    }

    return output;
}


module.exports = {
    convolve,
    convolveDelta
}
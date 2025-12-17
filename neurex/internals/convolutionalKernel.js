

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

const convolveDelta = (onGPU, delta_map, kernels, padding, strides) => {
    // delta_map: H x W x F (height, width, filters)
    // kernels: F x KH x KW x D (filters, kernel height, kernel width, depth)
    const rotated_Kernels = rotateKernel(kernels);
    const numFilters = kernels.length;
    const kernelHeight = rotated_Kernels[0].length;
    const kernelWidth = rotated_Kernels[0][0].length;
    const inputDepth = rotated_Kernels[0][0][0].length;
    const deltaHeight = delta_map.length;
    const deltaWidth = delta_map[0].length;

    // For 'full' padding, pad = kernel_size - 1
    const padH = kernelHeight - 1;
    const padW = kernelWidth - 1;

    // Output: height and width after full convolution
    const outHeight = deltaHeight + 2 * padH - (kernelHeight - 1);
    const outWidth = deltaWidth + 2 * padW - (kernelWidth - 1);

    // Output: H x W x D (match input to this conv layer)
    let output = [];
    for (let h = 0; h < deltaHeight + padH * 2 - (kernelHeight - 1); h++) {
        let row = [];
        for (let w = 0; w < deltaWidth + padW * 2 - (kernelWidth - 1); w++) {
            let depthArr = [];
            for (let d = 0; d < inputDepth; d++) {
                // For each input channel, sum over all filters
                let sum = 0;
                for (let f = 0; f < numFilters; f++) {
                    // Extract delta map for filter f (2D)
                    let delta2d = [];
                    for (let i = 0; i < deltaHeight; i++) {
                        delta2d.push(delta_map[i].map(cell => cell[f]));
                    }
                    // Extract rotated kernel for filter f, channel d (2D)
                    let kernel2d = [];
                    for (let i = 0; i < kernelHeight; i++) {
                        kernel2d.push([]);
                        for (let j = 0; j < kernelWidth; j++) {
                            kernel2d[i].push(rotated_Kernels[f][i][j][d]);
                        }
                    }
                    // Pad delta2d for 'full' convolution
                    let paddedDelta = applyPadding2D(delta2d, kernelHeight - 1, kernelWidth - 1);
                    // Perform 2D convolution at (h, w)
                    sum += conv2dAt(paddedDelta, kernel2d, h, w);
                }
                depthArr.push(sum);
            }
            row.push(depthArr);
        }
        output.push(row);
    }
    return output;
};

// Helper: 2D convolution at a single location (valid window)
function conv2dAt(input, kernel, h, w) {
    let kh = kernel.length;
    let kw = kernel[0].length;
    let sum = 0;
    for (let i = 0; i < kh; i++) {
        for (let j = 0; j < kw; j++) {
            let inVal = input[h + i] && input[h + i][w + j] !== undefined ? input[h + i][w + j] : 0;
            sum += inVal * kernel[i][j];
        }
    }
    return sum;
}

// Helper: Pad a 2D array with zeros
function applyPadding2D(input, padH, padW) {
    const inH = input.length;
    const inW = input[0].length;
    const out = [];
    for (let i = 0; i < inH + 2 * padH; i++) {
        let row = [];
        for (let j = 0; j < inW + 2 * padW; j++) {
            if (i < padH || i >= inH + padH || j < padW || j >= inW + padW) {
                row.push(0);
            } else {
                row.push(input[i - padH][j - padW]);
            }
        }
        out.push(row);
    }
    return out;
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
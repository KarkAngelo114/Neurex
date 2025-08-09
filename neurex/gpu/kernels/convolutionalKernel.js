const {gpu} = require('../gpu-init');

const convolve = (onGPU, filters, strides, input, kernels, biases, padding = "valid") => {
    if (onGPU) {
        // TODO
    }
    else {
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

function applyPadding(input, pad) {
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
}
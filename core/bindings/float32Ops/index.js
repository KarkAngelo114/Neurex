
exports.relu_float32 = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] > 0 ? output[i] : 0;
    }
    return output;
};

exports.sigmoid_float32 = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = 1 / (1 + Math.exp(-output[i]));
    }
    return output;
};

exports.tanh_float32 = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = Math.tanh(output[i]);
    }
    return output;
};

exports.softmax_float32 = (arr) => {
    const output = new Float32Array(arr);
    const maxVal = Math.max(...output);
    let sum = 0;

    for (let i = 0; i < output.length; i++) {
        output[i] = Math.exp(output[i] - maxVal);
        sum += output[i];
    }

    for (let i = 0; i < output.length; i++) {
        output[i] /= sum;
    }

    return output;
};

exports.linear_float32 = (arr) => {
    return new Float32Array(arr);
};

exports.drelu_float32 = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] > 0 ? 1 : 0;
    }
    return output;
};

exports.dsigmoid_float32 = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        const s = 1 / (1 + Math.exp(-output[i]));
        output[i] = s * (1 - s);
    }
    return output;
};

exports.dtanh_float32 = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        const t = Math.tanh(output[i]);
        output[i] = 1 - t * t;
    }
    return output;
};

exports.dsoftmax_float32 = (arr) => {

    return new Float32Array(arr.length).fill(1);
};

exports.dlinear_float32 = (arr) => {
    const output = new Float32Array(arr.length);
    output.fill(1);
    return output;
};




exports.MatMul_Float32 = (input, weights, biases, inputSize, outputSize) => {
    const z_values = new Float32Array(outputSize);

    // 1. Initialize with Biases (Faster than adding them in a separate loop later)
    z_values.set(biases);

    // 2. Perform Weighted Sum
    // We iterate through each input neuron
    for (let i = 0; i < inputSize; i++) {
        const inputVal = input[i];
        
        // Calculate the starting offset for this specific input neuron's weights
        const offset = i * outputSize;

        // Multiply the input by every weight connecting to output neurons
        for (let j = 0; j < outputSize; j++) {
            z_values[j] += inputVal * weights[offset + j];
        }
    }

    return z_values;
}

exports.DeltaMatMul_Float32 = (delta, weights, inputSize, outputSize) => {

    const prevDelta = new Float32Array(inputSize);

    // In a normal MatMul, we do: Input (inputSize) * Weights (inputSize x outputSize) = Output (outputSize)
    // In Delta MatMul, we do: Delta (outputSize) * Weights_Transposed (outputSize x inputSize) = PrevDelta (inputSize)

    for (let i = 0; i < inputSize; i++) {
        let sum = 0;
        const offset = i * outputSize;

        for (let j = 0; j < outputSize; j++) {
            // We multiply the j-th delta by the weight connecting input i to output j
            sum += weights[offset + j]  * delta[j];
        }
        prevDelta[i] = sum;
    }

    return prevDelta;
}

exports.computeWeightGradientsForWeightsInConnectedLayer_float32 = (activations, delta, weightGrads, inputSize, outputSize) => {
    const output = weightGrads;
    // We iterate through every connection
    for (let i = 0; i < inputSize; i++) {
        const inputVal = activations[i];
        
        // Calculate the row offset in the flat 1D array
        const offset = i * outputSize;

        for (let j = 0; j < outputSize; j++) {
            // weightGrads[index] += activation_i * delta_j
            output[offset + j] += inputVal * delta[j];
        }
    }

    return output;
}

exports.computeBiasGradsForConnected_Layer_float32 = (biasGrads, delta) => {
    const output = biasGrads;

    for (let i = 0; i < delta.length; i++) {
        output[i] += delta[i];
    }

    return output;
}

exports.scaleGrads_float32 = (grads, batchSize) => {
    const output = grads;

    for (let i = 0; i < grads.length; i++) {
        output[i] /= batchSize;
    }

    return output;
}

exports.ApplySGD_float32 = (params, grads, lr) => {
    const output = params;

    for (let i = 0; i < output.length; i++) {
        output[i] -= lr * grads[i];
    }

    return output;
}

exports.ApplyAdam_float32 = (params, grads, learning_rate, m, v, t, epsilon, beta1, beta2) => {
    const output = params;
    const output_M = m;
    const output_V = v;

    for (let i = 0; i < grads.length; i++) {
        let g = grads[i];

        m[i] = beta1 * m[i] + (1 - beta1) * g;
        v[i] = beta2 * v[i] + (1 - beta2) * g * g;

        let mHat = m[i] / (1.00 - Math.pow(beta1, t));
        let vHat = v[i] / (1.00 - Math.pow(beta2, t));

        output[i] -= learning_rate * mHat / (Math.sqrt(vHat) + epsilon);

    }

    return {
        params: output,
        m: output_M,
        v: output_V
        
    }

}

exports.ApplyPadding_Float32 = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => {
    const newH = inputH + padTop + padBottom;
    const newW = inputW + padLeft + padRight;
    const output = new Float32Array(newH * newW * channels);

    for (let i = 0; i < inputH; i++) {
        for (let j = 0; j < inputW; j++) {
            for (let c = 0; c < channels; c++) {
                const oldIdx = (i * inputW + j) * channels + c;
                const newIdx = ((i + padTop) * newW + (j + padLeft)) * channels + c;
                output[newIdx] = input[oldIdx];
            }
        }
    }
    return {
        data: output,
        shape: [newH, newW, channels]
    };
};


exports.Convolve_Float32 = ( input, kernels, biases, strides, outputH, outputW, num_filters, kernel_height, kernel_width, depth, inputH, inputW ) => {

    const output = new Float32Array(outputH * outputW * num_filters);

    for (let f = 0; f < num_filters; f++) {

        const bias = biases[f];

        for (let y = 0; y < outputH; y++) {
            for (let x = 0; x < outputW; x++) {

                let sum = 0;

                for (let ky = 0; ky < kernel_height; ky++) {
                    for (let kx = 0; kx < kernel_width; kx++) {
                        for (let c = 0; c < depth; c++) {

                            const inY = y * strides + ky;
                            const inX = x * strides + kx;

                            if (inY < inputH && inX < inputW) {

                                const inputIndex = ((inY * inputW + inX) * depth + c);

                                const kernelIndex = (((f * kernel_height + ky) * kernel_width + kx) * depth + c);

                                sum += input[inputIndex] * kernels[kernelIndex];
                            }
                        }
                    }
                }

                const outIndex = ((y * outputW + x) * num_filters + f);

                output[outIndex] = sum + bias;
            }
        }
    }

    return output;
};

exports.DilateDelta_Float32 = (input, shape, stride) => {
    const [H, W, C] = shape;
    const dilatedH = H * stride + (H - 1) * (stride - 1);
    const dilatedW = W * stride + (W - 1) * (stride - 1);
    const dilatedSize = dilatedH * dilatedW * C;
    const dilated = new Float32Array(dilatedSize);

    for (let c = 0; c < C; c++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const srcIdx = (h * W + w) * C + c;
                const dilatedHIdx = h * stride;
                const dilatedWIdx = w * stride;
                const dstIdx = (dilatedHIdx * dilatedW + dilatedWIdx) * C + c;
                dilated[dstIdx] = input[srcIdx] || 0;
            }
        }
    }

    return dilated;
};

exports.RotateKernels_Float32 = (kernels, F, KH, KW, D) => {
    const rotated = new Float32Array(kernels.length);

    for (let f = 0; f < F; f++) {
        for (let kh = 0; kh < KH; kh++) {
            for (let kw = 0; kw < KW; kw++) {
                for (let d = 0; d < D; d++) {
                    // Original Index
                    const oldIdx = (f * KH * KW * D) + (kh * KW * D) + (kw * D) + d;
                    
                    // Rotated Index (Flip KH and KW)
                    const newKh = KH - 1 - kh;
                    const newKw = KW - 1 - kw;
                    const newIdx = (f * KH * KW * D) + (newKh * KW * D) + (newKw * D) + d;
                    
                    rotated[newIdx] = kernels[oldIdx];
                }
            }
        }
    }
    return rotated;
};

/**
 * 
 * @param {Float32Array} padded 
 * @param {Array<Number>} padded_delta_shape - [Hp, Wp, C]
 * @param {Float32Array} rotatedKernels 
 * @param {Array<Number>} kernels_shape - [F, KH, KW, C]
 * @returns {Float32Array} output tensor [H, W, F]
 */
exports.ConvolveDelta_Float32 = (padded, padded_delta_shape, rotatedKernels, kernels_shape, oH, oW) => {

    const [Hp, Wp, C_in] = padded_delta_shape;
    const [F, KH, KW, C_k] = kernels_shape;

    // Match C++ logic
    const C = Math.min(C_in, C_k);

    // Infer output size (same as inputH, inputW in C++)
    const H = Hp - KH + 1;
    const W = Wp - KW + 1;

    const output = new Float32Array(oH * oW * F);

    // ---- Index helpers ----
    const idx3 = (h, w, c, W, C) => (h * W + w) * C + c;
    const idx4 = (f, kh, kw, c, KH, KW, C) => ((f * KH + kh) * KW + kw) * C + c;
    const idxOut = (h, w, f, W, F) => (h * W + w) * F + f;

    // ---- Convolution ----
    for (let f = 0; f < F; f++) {

        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {

                let sum = 0.0;

                for (let kh = 0; kh < KH; kh++) {
                    for (let kw = 0; kw < KW; kw++) {

                        const ph = h + kh;
                        const pw = w + kw;

                        for (let c = 0; c < C; c++) {

                            const inputVal = padded[idx3(ph, pw, c, Wp, C_in)];

                            const kernelVal = rotatedKernels[idx4(f, kh, kw, c, KH, KW, C_k)];

                            sum += inputVal * kernelVal;
                        }
                    }
                }

                output[idxOut(h, w, f, W, F)] = sum;
            }
        }
    }

    return output;
};

exports.computeBiasGradsForConv_Float32 = (grads, delta, outH, outW, numFilters) => {
    for (let f = 0; f < numFilters; f++) {
        let sum = 0;

        for (let h = 0; h < outH; h++) {
            for (let w = 0; w < outW; w++) {
                const idx = (h * outW + w) * numFilters + f;
                sum += delta[idx];
            }
        }

        grads[f] += sum;
    }

    return grads;
};

exports.computeKernelGradients_Float32 = (
    input, 
    delta,
    weightGrads,
    inputH, inputW, Cin,
    H, W, Cout,
    Kh, Kw
) => {

    const padH = Math.floor(Kh / 2);
    const padW = Math.floor(Kw / 2);

    for (let f = 0; f < Cout; f++) {
        for (let kh = 0; kh < Kh; kh++) {
            for (let kw = 0; kw < Kw; kw++) {
                for (let c = 0; c < Cin; c++) {

                    let sum = 0;

                    for (let h = 0; h < H; h++) {
                        for (let w = 0; w < W; w++) {

                            const inH = h + kh - padH;
                            const inW = w + kw - padW;

                            if (
                                inH >= 0 && inH < inputH &&
                                inW >= 0 && inW < inputW
                            ) {

                                const inputIndex =
                                    (inH * inputW + inW) * Cin + c;

                                const deltaIndex =
                                    (h * W + w) * Cout + f;

                                sum += input[inputIndex] * delta[deltaIndex];
                            }
                        }
                    }

                    const gradIndex = ((f * Kh + kh) * Kw + kw) * Cin + c;

                    weightGrads[gradIndex] += sum;
                }
            }
        }
    }

    return weightGrads;
}
const { getGlobalParams, replaceWeightParamByIndex } = require("../../../gpu/globals");

exports.Relu = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] > 0 ? output[i] : 0;
    }
    return output;
};

exports.Sigmoid = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = 1 / (1 + Math.exp(-output[i]));
    }
    return output;
};

exports.Tanh = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = Math.tanh(output[i]);
    }
    return output;
};

exports.Softmax = (arr) => {
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

exports.Linear = (arr) => {
    return new Float32Array(arr);
};

exports.DReLu = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] > 0 ? 1 : 0;
    }
    return output;
};

exports.DSigmoid = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        const s = 1 / (1 + Math.exp(-output[i]));
        output[i] = s * (1 - s);
    }
    return output;
};

exports.DTanh = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        const t = Math.tanh(output[i]);
        output[i] = 1 - t * t;
    }
    return output;
};

exports.DSoftmax = (arr) => {

    return new Float32Array(arr.length).fill(1);
};

exports.DLinear = (arr) => {
    const output = new Float32Array(arr.length);
    output.fill(1);
    return output;
};


exports.getEmbeddings = (tokenVector, embeddingDim, pointer, outputTemplatePointer) => {
    const {globalWeights, globalOutputTensorTemplate} = getGlobalParams();

    const lookup = globalWeights[pointer];
    const output = globalOutputTensorTemplate[outputTemplatePointer];

    // helper function
    const getRow = (tokenID) => {
        const start = tokenID * embeddingDim;

        return lookup.subarray(start, start + embeddingDim);
    }

    const sequence_length = tokenVector.length;

    for (let i = 0; i < sequence_length; i++) {
        const row = getRow(tokenVector[i]);

        output.set(row, i * embeddingDim);
    }

    return output;
}

exports.returnEmbeddings = (activation_outputs, delta, weightGrads, dim) => {
    const embeddingDim = dim;
    
    for (let i = 0; i < activation_outputs.length; i++) {
        const tokenId = activation_outputs[i];
    
        if (tokenId === 0) continue;  // skip whos IDs are reserved index which is 0s <PAD>
    
        const gradOffset = tokenId * embeddingDim;
        const deltaOffset = i * embeddingDim;
    
        for (let d = 0; d < embeddingDim; d++) {
            weightGrads[gradOffset + d] += delta[deltaOffset + d];
        }
    }
    
    return weightGrads;
}

exports.MatMul = (input, inputSize, outputSize, pointer, outputTemplatePointer) => {

    /**
     * since there's no weights and biases being passed to this function, we use the pointer to reference the parameters
     */

    const {globalWeights, globalBiases, globalOutputTensorTemplate} = getGlobalParams();
    
    const z_values = globalOutputTensorTemplate[outputTemplatePointer]; // use the output template pointer to get the corresponding pre-allocated output tensor

    
    // 1. Initialize with Biases (Faster than adding them in a separate loop later)
    z_values.set(globalBiases[pointer]);

    // 2. Perform Weighted Sum
    // We iterate through each input neuron
    for (let i = 0; i < inputSize; i++) {
        const inputVal = input[i];
        
        // Calculate the starting offset for this specific input neuron's weights
        const offset = i * outputSize;

        // Multiply the input by every weight connecting to output neurons
        for (let j = 0; j < outputSize; j++) {
            z_values[j] += inputVal * globalWeights[pointer][offset + j];
        }
    }

    return z_values;
}

exports.DeltaMatMul = (delta, inputSize, outputSize, pointer) => {
    const { globalWeights } = getGlobalParams();

    const prevDelta = new Float32Array(inputSize);

    // In a normal MatMul, we do: Input (inputSize) * Weights (inputSize x outputSize) = Output (outputSize)
    // In Delta MatMul, we do: Delta (outputSize) * Weights_Transposed (outputSize x inputSize) = PrevDelta (inputSize)

    for (let i = 0; i < inputSize; i++) {
        let sum = 0;
        const offset = i * outputSize;

        for (let j = 0; j < outputSize; j++) {
            // We multiply the j-th delta by the weight connecting input i to output j
            sum += globalWeights[pointer][offset + j]  * delta[j];
        }
        prevDelta[i] = sum;
    }

    return prevDelta;
}

exports.computeWeightGradientsForWeightsInConnectedLayer = (activations, delta, weightGrads, inputSize, outputSize) => {
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

exports.computeBiasGradsForConnected_Layer = (biasGrads, delta) => {
    const output = biasGrads;

    for (let i = 0; i < delta.length; i++) {
        output[i] += delta[i];
    }

    return output;
}

exports.scaleGrad = (grads, batchSize) => {
    const output = grads;

    for (let i = 0; i < grads.length; i++) {
        output[i] /= batchSize;
    }

    return output;
}

exports.SGD = (params, grads, lr) => {
    const output = params;

    for (let i = 0; i < output.length; i++) {
        output[i] -= lr * grads[i];
    }

    return output;
}

exports.Adam = (params, grads, m, v, t, learning_rate, beta1, beta2, epsilon) => {
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

exports.ApplyPadding = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => {
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

/**
 * 
 * @param {Float32Array} input 
 * @param {Number} strides 
 * @param {Array<Number>} outputShape 
 * @param {Array<Number>} kernelShape 
 * @param {Array<Number>} inputShape 
 * @param {Number} pointer 
 * @param {Number} outputTemplatePointer 
 * @returns 
 */
exports.Convolve = (input, strides, outputShape, kernelShape, inputShape, pointer) => {

    const [numFilters, kernelH, kernelW, depth] = kernelShape;
    const [inputH, inputW] = inputShape;
    const [outputH, outputW] = outputShape;

    const { globalWeights, globalBiases } = getGlobalParams();

    const weights = globalWeights[pointer];
    const biases = globalBiases[pointer];

    const output = new Float32Array(outputH * outputW * numFilters);

    const kernelSize = kernelH * kernelW * depth;

    for (let y = 0; y < outputH; y++) {

        const baseY = y * strides;

        for (let x = 0; x < outputW; x++) {

            const baseX = x * strides;

            const outBase = (y * outputW + x) * numFilters;

            for (let f = 0; f < numFilters; f++) {

                let sum = biases[f];

                const filterOffset = f * kernelSize;

                for (let ky = 0; ky < kernelH; ky++) {

                    const inY = baseY + ky;

                    if (inY >= inputH) continue;

                    for (let kx = 0; kx < kernelW; kx++) {

                        const inX = baseX + kx;

                        if (inX >= inputW) continue;

                        const inputBase = (inY * inputW + inX) * depth;

                        const kernelBase = filterOffset + (ky * kernelW + kx) * depth;

                        let c = 0;

                        for (; c <= depth - 4; c += 4) {
                            sum += input[inputBase + c] * weights[kernelBase + c];
                            sum += input[inputBase + c + 1] * weights[kernelBase + c + 1];
                            sum += input[inputBase + c + 2] * weights[kernelBase + c + 2];
                            sum += input[inputBase + c + 3] * weights[kernelBase + c + 3];
                        }

                        for (; c < depth; c++) {
                            sum += input[inputBase + c] * weights[kernelBase + c];
                        }
                    }
                }

                output[outBase + f] = sum;
            }
        }
    }

    return output;
};


exports.DilateInput = (input, shape, stride) => {
    const [H, W, C] = shape;
    const dilatedH = (H - 1) * stride + 1;
    const dilatedW = (W - 1) * stride + 1;
    
    const dilatedSize = dilatedH * dilatedW * C;
    const dilated = new Float32Array(dilatedSize);

    for (let c = 0; c < C; c++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const srcIdx = (h * W + w) * C + c;
                const dilatedHIdx = h * stride;
                const dilatedWIdx = w * stride;
                const dstIdx = (dilatedHIdx * dilatedW + dilatedWIdx) * C + c;
                dilated[dstIdx] = input[srcIdx];
            }
        }
    }

    return {
        data: dilated,
        dilatedHeight: dilatedH,
        dilatedWidth: dilatedW
    };
};

const RotateKernels = (F, KH, KW, D, pointer) => {
    const {globalWeights} = getGlobalParams();
    const rotated = new Float32Array(globalWeights[pointer].length);

    for (let f = 0; f < F; f++) {
        for (let kh = 0; kh < KH; kh++) {
            for (let kw = 0; kw < KW; kw++) {
                for (let d = 0; d < D; d++) {
                    const oldIdx = (f * KH * KW * D) + (kh * KW * D) + (kw * D) + d;
                    const newKh = KH - 1 - kh;
                    const newKw = KW - 1 - kw;
                    const newIdx = (f * KH * KW * D) + (newKh * KW * D) + (newKw * D) + d;
                    
                    rotated[newIdx] = globalWeights[pointer][oldIdx];
                }
            }
        }
    }
    // Return the rotated array for temporary use[cite: 1]
    return rotated; 
};

/**
 * 
 * @param {Float32Array} input 
 * @param {Array<Number>} delta_shape 
 * @param {Array<Number>} kernels_shape 
 * @param {Array<Number>} outputShape 
 * @param {Number} pointer 
 * @param {Number} stride 
 * @returns 
 */
exports.ConvolveDelta = (input, delta_shape, kernels_shape, outputShape, pointer, stride) => {

    const [Hp, Wp, C_in] = delta_shape;
    const [F, KH, KW, C_k] = kernels_shape;
    const [oH, oW] = outputShape;

    // rotate kernels
    const rotated_kernel = RotateKernels(F, KH, KW, C_k, pointer);

    // Infer output size (same as inputH, inputW in C++)
    const H = Hp - KH + 1;
    const W = Wp - KW + 1;

    const output = new Float32Array(oH * oW * C_k);

    // ---- Convolution ----
    for (let c_out = 0; c_out < C_k; c_out++) {     // output channel = previous depth
        for (let h = 0; h < oH; h++) {
            for (let w = 0; w < oW; w++) {
                let sum = 0;
                for (let kh = 0; kh < KH; kh++) {
                    for (let kw = 0; kw < KW; kw++) {
                        const ph = h * stride + kh;
                        const pw = w * stride + kw;
                        const baseIdx = (ph * Wp + pw) * C_in;
                        const kernelBase = ((kh * KW + kw) * F) * C_k + c_out;

                        let f = 0;
                        for (; f <= F - 4; f += 4) {
                            sum += input[baseIdx + f] * rotated_kernel[f * C_k + kernelBase];
                            sum += input[baseIdx + f + 1] * rotated_kernel[(f + 1) * C_k + kernelBase];
                            sum += input[baseIdx + f + 2] * rotated_kernel[(f + 2) * C_k + kernelBase];
                            sum += input[baseIdx + f + 3] * rotated_kernel[(f + 3) * C_k + kernelBase];
                        }

                        for (; f < F; f++) {
                            const padIdx = baseIdx + f;
                            const kernelIdx = ((f * KH + kh) * KW + kw) * C_k + c_out;
                            sum += input[padIdx] * rotated_kernel[kernelIdx];
                        }
                    }
                }
                output[(h * oW + w) * C_k + c_out] = sum;
            }
        }
    }
    return output;
};

exports.computeBiasGradsForConv = (grads, delta, outH, outW, numFilters) => {
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

/**
 * 
 * @param {Float32Array} input 
 * @param {Float32Array} delta 
 * @param {Float32Array} weightGrads 
 * @param {Array<Number>} inputShape 
 * @param {Array<Number>} outputShape 
 * @param {Array<Number>} kernelSize 
 * @param {Array<Number>} stride 
 * @returns 
 */
exports.computeKernelGradients = (input, delta, weightGrads, inputShape, outputShape, kernelSize, stride) => {

    const [inputH, inputW, Cin] = inputShape;
    const [H, W, Cout] = outputShape; 
    const [Kh, Kw] = kernelSize;

    const padH = Math.floor(Kh / 2);
    const padW = Math.floor(Kw / 2);

    for (let f = 0; f < Cout; f++) {
        for (let kh = 0; kh < Kh; kh++) {
            for (let kw = 0; kw < Kw; kw++) {
                const kernelRowOffset = (f * Kh + kh) * Kw + kw;

                let c = 0;
                for (; c <= Cin - 4; c += 4) {
                    let sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

                    for (let h = 0; h < H; h++) {
                        for (let w = 0; w < W; w++) {
                            const inH = (h * stride) + kh - padH;
                            const inW = (w * stride) + kw - padW;

                            if (inH >= 0 && inH < inputH && inW >= 0 && inW < inputW) {
                                const baseInputIndex = (inH * inputW + inW) * Cin;
                                const deltaIndex = (h * W + w) * Cout + f;
                                const deltaVal = delta[deltaIndex];

                                sum0 += input[baseInputIndex + c] * deltaVal;
                                sum1 += input[baseInputIndex + c + 1] * deltaVal;
                                sum2 += input[baseInputIndex + c + 2] * deltaVal;
                                sum3 += input[baseInputIndex + c + 3] * deltaVal;
                            }
                        }
                    }

                    weightGrads[kernelRowOffset * Cin + c] += sum0;
                    weightGrads[kernelRowOffset * Cin + c + 1] += sum1;
                    weightGrads[kernelRowOffset * Cin + c + 2] += sum2;
                    weightGrads[kernelRowOffset * Cin + c + 3] += sum3;
                }

                // Process remaining channels
                for (; c < Cin; c++) {
                    let sum = 0;

                    for (let h = 0; h < H; h++) {
                        for (let w = 0; w < W; w++) {
                            const inH = (h * stride) + kh - padH;
                            const inW = (w * stride) + kw - padW;

                            if (inH >= 0 && inH < inputH && inW >= 0 && inW < inputW) {
                                const inputIndex = (inH * inputW + inW) * Cin + c;
                                const deltaIndex = (h * W + w) * Cout + f;
                                sum += input[inputIndex] * delta[deltaIndex];
                            }
                        }
                    }

                    const gradIndex = kernelRowOffset * Cin + c;
                    weightGrads[gradIndex] += sum;
                }
            }
        }
    }

    return weightGrads;
}


exports.MaxPooling = (arr, pool_size, inputShape, outputShape, strides, outputTemplatePointer) => {
    const {globalOutputTensorTemplate} = getGlobalParams();
    const [poolH, poolW] = pool_size;
    const [inputH, inputW, inputD] = inputShape;
    const [outputH, outputW, outputD] = outputShape;

    const output = globalOutputTensorTemplate[outputTemplatePointer];
    const maxIdexes = new Int32Array(outputH * outputW * outputD);

    for (let d = 0; d < inputD; d++) {
        for (let i = 0; i < outputH; i++) {
            for (let j = 0; j < outputW; j++) {
                let maxVal = -Infinity;
                let maxIdx = -1;
                // Define the window boundaries based on strides
                const startH = i * strides;
                const startW = j * strides;

                for (let ph = 0; ph < poolH; ph++) {
                    for (let pw = 0; pw < poolW; pw++) {
                        const currH = startH + ph;
                        const currW = startW + pw;

                        // Check bounds to handle cases where window might exceed input dimensions
                        if (currH < inputH && currW < inputW) {
                            // Calculate index in the flattened 1D array
                            const idx = (currH * inputW * inputD) + (currW * inputD) + d;
                            const val = arr[idx];
                            if (val > maxVal) {
                                maxVal = val;
                                maxIdx = idx;
                            };
                        }
                    }
                }
                // Set the max value in the output array
                const outIdx = (i * outputW * outputD) + (j * outputD) + d;
                output[outIdx] = maxVal === -Infinity ? 0 : maxVal;
                maxIdexes[outIdx] = maxIdx;
            }
        }
    }
    return {
        output: output,
        maxIndices: maxIdexes
    };
}

exports.MaxPoolDelta = (delta, indices, H, W, D) => {
    const output = new Float32Array(H * W * D);

    for (let i = 0; i < indices.length; i++) {
        let idx = indices[i];
        output[idx] += delta[i];
    }

    return output;

}

exports.element_wise_mul = (arr1, arr2) => {
    let output = new Float32Array(arr1.length);

    for (let i = 0; i < arr1.length; i++) {
        output[i] = arr1[i] * arr2[i];
    }

    return output;
}

exports.scaleDiff = (arr1, arr2, arr3) => {
    let output = new Float32Array(arr1.length);

    for (let i = 0; i < output.length; i++) {
        output[i] = (arr1[i] - arr2[i]) * arr3[i];
    }

    return output;
}

exports.element_wise_sub = (arr1, arr2) => {
    let output = new Float32Array(arr1.length);

    for (let i = 0; i < output.length; i++) {
        output[i] = arr1[i] - arr2[i];
    }

    return output;
}

exports.mse = (predictions, actuals) => {
    let occurrence = predictions.length;
    let sum = 0;
    for (let i = 0; i < occurrence; i++) {
        let difference = predictions[i] - actuals[i];
        sum += difference * difference;
    }

    return sum / occurrence;
}   

exports.mae = (predictions, actuals) => {
    let occurrence = predictions.length;
    let sum = 0;
    for (let i = 0; i < occurrence; i++) {
        sum += Math.abs(predictions[i] - actuals[i]);
    }

    return sum / occurrence;
}

exports.categorical_cross_entropy = (predictions, actuals, epsilon) => {
    let loss = 0;
    for (let i = 0; i < predictions.length; i++) {
        loss -= actuals[i] * Math.log(Math.max(predictions[i], epsilon));
    }

    return loss;
}


exports.sparse_categorical_cross_entropy = (predictions, actuals, epsilon) => {
    const p = Math.max(predictions[actuals[0]], epsilon); // actuals being passed here can be use to index the predicted output because the actuals are like this: [0], [4], [1], and so on
    return -Math.log(p);
}

exports.binary_cross_entropy = (predictions, actuals, epsilon) => {
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
        const p = Math.max(Math.min(predictions[i], 1 - epsilon), epsilon);
        sum -= actuals[i] * Math.log(p) + (1 - actuals[i]) * Math.log(1 - p);
    }
    return sum / predictions.length;
}
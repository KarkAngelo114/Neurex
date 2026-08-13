const { getGlobalParams, replaceWeightParamByIndex } = require("../../../gpu/globals");

const Relu = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] > 0 ? output[i] : 0;
    }
    return output;
};

const Sigmoid = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = 1 / (1 + Math.exp(-output[i]));
    }
    return output;
};

const Tanh = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = Math.tanh(output[i]);
    }
    return output;
};

const Softmax = (arr) => {
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

const Linear = (arr) => {
    return new Float32Array(arr);
};

const DReLu = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        output[i] = output[i] > 0 ? 1 : 0;
    }
    return output;
};

const DSigmoid = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        const s = 1 / (1 + Math.exp(-output[i]));
        output[i] = s * (1 - s);
    }
    return output;
};

const DTanh = (arr) => {
    const output = new Float32Array(arr);
    for (let i = 0; i < output.length; i++) {
        const t = Math.tanh(output[i]);
        output[i] = 1 - t * t;
    }
    return output;
};

const DSoftmax = (arr) => {

    return new Float32Array(arr.length).fill(1);
};

const DLinear = (arr) => {
    const output = new Float32Array(arr.length);
    output.fill(1);
    return output;
};

const getEmbeddings = (tokenVector, embeddingDim, lookup, outputTemplatePointer) => {
    const {globalOutputTensorTemplate} = getGlobalParams();

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

const returnEmbeddings = (activation_outputs, delta, weightGrads, dim) => {
    
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


const MatMul = (input, inputSize, outputSize, weights, biases, outputTemplatePointer) => {
    const { globalOutputTensorTemplate } = getGlobalParams();

    const output = globalOutputTensorTemplate[outputTemplatePointer];

    output.set(biases);

    for (let i = 0; i < inputSize; i++) {
        const inputVal = input[i];

        const rowStart = i * outputSize;
        const rowEnd = rowStart + outputSize;
        const weightRow = weights.subarray(rowStart, rowEnd);

        for (let j = 0; j < outputSize; j++) {
            output[j] += inputVal * weightRow[j];
        }
    }

    return output;
};

const DeltaMatMul = (delta, inputSize, outputSize, weights) => {
    const prevDelta = new Float32Array(inputSize);

    for (let i = 0; i < inputSize; i++) {
        const start = i * outputSize;
        const end = start + outputSize;

        const weight = weights.subarray(start, end);
        let sum = 0;

        for (let j = 0; j < outputSize; j++) {
            sum += delta[j] * weight[j];
        }

        prevDelta[i] = sum;
    }

    return prevDelta;
}

const computeWeightGradientsForWeightsInConnectedLayer = (activations, delta, weightGrads, inputSize, outputSize) => {
    for (let i = 0; i < inputSize; i++) {
        const inputVal = activations[i];

        const rowStart = i * outputSize;
        const rowEnd = rowStart + outputSize;
        const gradRow = weightGrads.subarray(rowStart, rowEnd);

        for (let j = 0; j < outputSize; j++) {
            gradRow[j] += inputVal * delta[j];
        }
    }

    return weightGrads;
}

const computeBiasGradsForConnected_Layer = (biasGrads, delta) => {
    const output = biasGrads;

    for (let i = 0; i < delta.length; i++) {
        output[i] += delta[i];
    }

    return output;
}

const scaleGrad = (grads, batchSize) => {
    const output = grads;

    for (let i = 0; i < grads.length; i++) {
        output[i] /= batchSize;
    }

    return output;
}

const gradientClipping = (grads, threshold) => {
    let norm = 0;
    
    for (let i = 0; i < grads.length; i++) {
        norm += grads[i] * grads[i];
    }
    norm = Math.sqrt(norm);

    if (norm > threshold) {
        let scalingValue = threshold / norm;
        for (let i = 0; i < grads.length; i++) {
            grads[i] *= scalingValue;
        }
    }

    return grads;
}

const SGD = (params, grads, velocity, lr, momentum = 0.9) => {

    for (let i = 0; i < params.length; i++) {

        velocity[i] = momentum * velocity[i] + grads[i];

        params[i] -= lr * velocity[i];
    }

    return {
        params: params,
        velocity: velocity
    };
};

const Adam = (params, grads, m, v, t, learning_rate, beta1, beta2, epsilon) => {
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

const ApplyPadding = (input, inputH, inputW, channels, padTop, padBottom, padLeft, padRight) => {
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
 * @param {Float32Array} weights
 * @param {Float32Array} biases
 * @returns 
 */
const Convolve = (input, strides, outputShape, kernelShape, inputShape, weights, biases) => {

    
    const [numFilters, kernelH, kernelW, depth] = kernelShape;
    const [inputH, inputW] = inputShape;
    const [outputH, outputW] = outputShape;

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


const DilateInput = (input, shape, stride) => {
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

const RotateKernels = (F, KH, KW, D, weights) => {
    const rotated = new Float32Array(weights.length);

    for (let f = 0; f < F; f++) {
        for (let kh = 0; kh < KH; kh++) {
            for (let kw = 0; kw < KW; kw++) {
                for (let d = 0; d < D; d++) {
                    const oldIdx = (f * KH * KW * D) + (kh * KW * D) + (kw * D) + d;
                    const newKh = KH - 1 - kh;
                    const newKw = KW - 1 - kw;
                    const newIdx = (f * KH * KW * D) + (newKh * KW * D) + (newKw * D) + d;
                    
                    rotated[newIdx] = weights[oldIdx];
                }
            }
        }
    }
    // Return the rotated array for temporary use
    return rotated; 
};

/**
 * 
 * @param {Float32Array} input 
 * @param {Array<Number>} delta_shape 
 * @param {Array<Number>} kernels_shape 
 * @param {Array<Number>} outputShape 
 * @param {Float32Array} weights 
 * @param {Number} stride 
 * @returns 
 */
const ConvolveDelta = (input, delta_shape, kernels_shape, outputShape, weights, stride) => {

    const [Hp, Wp, C_in] = delta_shape;
    const [F, KH, KW, C_k] = kernels_shape;
    const [oH, oW] = outputShape;

    // rotate kernels
    const rotated_kernel = RotateKernels(F, KH, KW, C_k, weights);

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

const computeBiasGradsForConv = (grads, delta, outH, outW, numFilters) => {
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
const computeKernelGradients = (input, delta, weightGrads, inputShape, outputShape, kernelSize, stride) => {

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

const MaxPooling = (arr, pool_size, inputShape, outputShape, strides, outputTemplatePointer) => {
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

const MaxPoolDelta = (delta, indices, H, W, D) => {
    const output = new Float32Array(H * W * D);

    for (let i = 0; i < indices.length; i++) {
        let idx = indices[i];
        output[idx] += delta[i];
    }

    return output;

}

const element_wise_mul = (arr1, arr2) => {
    let output = new Float32Array(arr1.length);

    for (let i = 0; i < arr1.length; i++) {
        output[i] = arr1[i] * arr2[i];
    }

    return output;
}

const scaleDiff = (arr1, arr2, arr3) => {
    let output = new Float32Array(arr1.length);
    const scale = 2.0 / output.length;

    for (let i = 0; i < output.length; i++) {
        output[i] = (arr1[i] - arr2[i]) * arr3[i] * scale;
    }

    return output;
}

const element_wise_sub = (arr1, arr2) => {
    let output = new Float32Array(arr1.length);

    for (let i = 0; i < output.length; i++) {
        output[i] = arr1[i] - arr2[i];
    }

    return output;
}

const mse = (predictions, actuals) => {
    let occurrence = predictions.length;
    let sum = 0;
    for (let i = 0; i < occurrence; i++) {
        let difference = predictions[i] - actuals[i];
        sum += difference * difference;
    }

    return sum / occurrence;
}   

const mae = (predictions, actuals) => {
    let occurrence = predictions.length;
    let sum = 0;
    for (let i = 0; i < occurrence; i++) {
        sum += Math.abs(predictions[i] - actuals[i]);
    }

    return sum / occurrence;
}

const categorical_cross_entropy = (predictions, actuals, epsilon) => {
    let loss = 0;
    for (let i = 0; i < predictions.length; i++) {
        loss -= actuals[i] * Math.log(Math.max(predictions[i], epsilon));
    }

    return loss;
}

const sparse_categorical_cross_entropy = (predictions, actuals, epsilon) => {
    const p = Math.max(predictions[actuals[0]], epsilon); // actuals being passed here can be use to index the predicted output because the actuals are like this: [0], [4], [1], and so on
    return -Math.log(p);
}

const binary_cross_entropy = (predictions, actuals, epsilon) => {
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
        const p = Math.max(Math.min(predictions[i], 1 - epsilon), epsilon);
        sum -= actuals[i] * Math.log(p) + (1 - actuals[i]) * Math.log(1 - p);
    }
    return sum / predictions.length;
}

const recurrentMatMul = (input, prevHiddenState,  inputWeightShape, recurrentWeightShape, weights, biases, outputTemplatePointer) => {
    const { globalOutputTensorTemplate } = getGlobalParams();
    // The weights were concatenated during initialization as:
    // [input_weights..., recurrent_weights...]
    const inputSize = inputWeightShape[0];
    const units = inputWeightShape[1];
    const range_input_weights = inputSize * units;
    const output = globalOutputTensorTemplate[outputTemplatePointer];

    const input_weights = weights.subarray(0, range_input_weights);
    const recurrent_weights = weights.subarray(range_input_weights, range_input_weights + recurrentWeightShape[0] * recurrentWeightShape[1]);

    for (let j = 0; j < units; j++) {
        let z = biases[j];

        for (let i = 0; i < inputSize; i++) {
            z += input[i] * input_weights[i * units + j];
        }

        for (let h = 0; h < units; h++) {
            z += prevHiddenState[h] * recurrent_weights[h * units + j];
        }

        output[j] = z;
    }

    return output;
}

const recurrentTimeDelta = (delta, inputWeightShape, recurrentWeightShape, weightParams) => {
    // input weight shape [feature_size, units]
    // recurrent weight shape [units, units]
    const a = inputWeightShape[0];
    const b = inputWeightShape[1];
    const c = recurrentWeightShape[0];
    const d = recurrentWeightShape[1];

    const offset = a * b;
    const length = c * d;

    const weights = weightParams.subarray(offset, offset + length); // get the array of recurrent weights
    
    const prevDelta = new Float32Array(delta);


    for (let i = 0; i < c; i++) {
        let sum = 0;
        const offset = i * c;

        for (let j = 0; j < d; j++) {

            sum += weights[offset + j]  * delta[j];
        }
        prevDelta[i] = sum;
    }

    return prevDelta;
}

const recurrentWeightGradsAccumulation = (activation_outputs, deltas, hiddenStates, deltaTs, weightGrads, weightShape, sequenceLength) => {
    let [featureSize, units] = weightShape;
    let output = weightGrads;

    const totalInputWeights = featureSize * units; // offset where recurrent-weight block starts

    for (let t = 0; t < sequenceLength; t++) {
        const x_t = activation_outputs.subarray(t * featureSize, (t + 1) * featureSize);
        const h_prev = t === 0 ? new Float32Array(units) : hiddenStates[t - 1];
        const delta_t = deltaTs[t];

        // dL/dW_x += outer(x_t, delta_t)      -- W_x is [featureSize, units], row-major
        for (let i = 0; i < featureSize; i++) {
            const xi = x_t[i];
            const rowOffset = i * units;
            for (let j = 0; j < units; j++) {
                output[rowOffset + j] += xi * delta_t[j]
            };
        }

        // dL/dW_h += outer(h_prev, delta_t)   -- W_h is [units, units], stored right after W_x
        for (let i = 0; i < units; i++) {
            const hi = h_prev[i];
            const rowOffset = totalInputWeights + i * units;
            for (let j = 0; j < units; j++) {
                output[rowOffset + j] += hi * delta_t[j];
            }
        }
    }

    return output;

}

const recurrentBiasGradsAccumulation = (biasGrads, deltaTs, sequenceLength, units) => {
    let output = biasGrads;

    for (let t = 0; t < sequenceLength; t++) {
        const delta_t = deltaTs[t];

        for (let j = 0; j < units; j++) {
            output[j] += delta_t[j];
        }
    }

    return output;
}

const transConv = (input, inputShape, outputShape, strides, filters, weightShape, weights, biases, outputTemplatePointer) => {
    const { globalOutputTensorTemplate } = getGlobalParams();

    const output = globalOutputTensorTemplate[outputTemplatePointer] || new Float32Array();
    const [iH, iW, iD] = inputShape;
    const [oH, oW, oD] = outputShape;
    const [f, kh, kw, d] = weightShape;

    // Sanity checks
    if (d !== iD) {
        throw new Error(`TransConv: weight input depth (${d}) != input depth (${iD})`);
    }

    if (f !== oD) {
        throw new Error(`TransConv: number of filters (${f}) != output depth (${oD})`);
    }

    if (filters !== f) {
        throw new Error(`TransConv: filters (${filters}) != weightShape[0] (${f})`);
    }

    // Clear output first (just in case)
    output.fill(0);

    const padH = Math.max(0, (iH - 1) * strides + kh - oH);
    const padW = Math.max(0,(iW - 1) * strides + kw - oW);
    const padTop = Math.floor(padH / 2);
    const padLeft = Math.floor(padW / 2);

    // Flat index helpers.
    const inputIndex = (y, x, c) => (y * iW + x) * iD + c;
    const outputIndex = (y, x, c) => (y * oW + x) * f + c;
    const weightIndex = (filter, ky, kx, c) =>(((filter * kh) + ky) * kw + kx) * d + c;

    for (let iy = 0; iy < iH; iy++) {
        for (let ix = 0; ix < iW; ix++) {

            const inputBase = (iy * iW + ix) * iD;

            for (let ky = 0; ky < kh; ky++) {

                const oy = iy * strides + ky - padTop;

                // Kernel row falls outside output.
                if (oy < 0 || oy >= oH) continue;

                for (let kx = 0; kx < kw; kx++) {

                    const ox = ix * strides + kx - padLeft;

                    // Kernel column falls outside output.
                    if (ox < 0 || ox >= oW) continue;

                    const outputBase = (oy * oW + ox) * f;

                    /*
                     * For every output filter, accumulate the
                     * input channels multiplied by the kernel.
                     */
                    for (let filter = 0; filter < f; filter++) {

                        let sum = 0;

                        const weightBase = ((filter * kh + ky) * kw + kx) * d;

                        for (let c = 0; c < d; c++) {
                            sum += input[inputBase + c] * weights[weightBase + c];
                        }

                        output[outputBase + filter] += sum;
                    }
                }
            }
        }
    }

    /*
     * Bias is added ONCE per output element, after all
     * input/kernel contributions have been accumulated.
     */
    for (let y = 0; y < oH; y++) {
        for (let x = 0; x < oW; x++) {

            const outputBase = (y * oW + x) * f;

            for (let filter = 0; filter < f; filter++) {
                output[outputBase + filter] += biases[filter];
            }
        }
    }

    return output;
};

const transConvBackward = (delta, inputShape, outputShape, strides, filters, weightShape, weights) => {

    const [iH, iW, iD] = inputShape;
    const [oH, oW, oD] = outputShape;
    const [f, kh, kw, d] = weightShape;

    // Sanity checks
    if (d !== iD) {
        throw new Error(`TransConvDelta: weight input depth (${d}) != input depth (${iD})`);
    }

    if (f !== oD) {
        throw new Error(`TransConvDelta: number of filters (${f}) != output depth (${oD})`);
    }

    if (filters !== f) {
        throw new Error(`TransConvDelta: filters (${filters}) != weightShape[0] (${f})`);
    }

    const deltaInput = new Float32Array(iH * iW * iD);

    const padH = Math.max(0, (iH - 1) * strides + kh - oH);

    const padW = Math.max(0, (iW - 1) * strides + kw - oW);

    const padTop = Math.floor(padH / 2);
    const padLeft = Math.floor(padW / 2);

    const deltaInputIndex = (y, x, c) => (y * iW + x) * iD + c;
    const deltaOutputIndex = (y, x, f) => (y * oW + x) * oD + f;

    const weightIndex = (f, ky, kx, c) => (((f * kh) + ky) * kw + kx) * d + c;

    for (let iy = 0; iy < iH; iy++) {
        for (let ix = 0; ix < iW; ix++) {
            for (let ky = 0; ky < kh; ky++) {
                const oy = iy * strides + ky - padTop;

                if (oy < 0 || oy >= oH) continue;

                for (let kx = 0; kx < kw; kx++) {

                    const ox = ix * strides + kx - padLeft;

                    if (ox < 0 || ox >= oW) continue;

                    for (let filter = 0; filter < f; filter++) {

                        const deltaY = delta[deltaOutputIndex(oy, ox, filter)];
                        const weightBase = ((filter * kh + ky) * kw + kx) * d;
                        const inputBase = (iy * iW + ix) * iD;

                        for (let c = 0; c < d; c++) {

                            deltaInput[inputBase + c] += deltaY * weights[weightBase + c];
                        }
                    }
                }
            }
        }
    }

    return deltaInput;
}

const accumulateKernelGradsForTransConv = (activation_outputs, deltas, weightGrads, strides, filters, inputShape, outputShape, weightShape) => {
    const [iH, iW, iD] = inputShape;
    const [oH, oW, oD] = outputShape;
    const [f, kh, kw, d] = weightShape;

    const padH = Math.max(0, (iH - 1) * strides + kh - oH);
    const padW = Math.max(0, (iW - 1) * strides + kw - oW);
    const padTop = Math.floor(padH / 2);
    const padLeft = Math.floor(padW / 2);

    for (let iy = 0; iy < iH; iy++) {
        for (let ix = 0; ix < iW; ix++) {
            const inputBase = (iy * iW + ix) * iD;

            for (let ky = 0; ky < kh; ky++) {
                const oy = iy * strides + ky - padTop;
                if (oy < 0 || oy >= oH) continue;

                for (let kx = 0; kx < kw; kx++) {
                    const ox = ix * strides + kx - padLeft;
                    if (ox < 0 || ox >= oW) continue;

                    const deltaBase = (oy * oW + ox) * filters;

                    for (let filter = 0; filter < filters; filter++) {
                        const deltaVal = deltas[deltaBase + filter];
                        const gradBase = ((filter * kh + ky) * kw + kx) * iD;

                        for (let c = 0; c < iD; c++) {
                            weightGrads[gradBase + c] += activation_outputs[inputBase + c] * deltaVal;
                        }
                    }
                }
            }
        }
    }

    return weightGrads;
}


module.exports = {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
    DReLu,
    DSigmoid,
    DTanh,
    DSoftmax,
    DLinear,
    getEmbeddings,
    returnEmbeddings,
    MatMul,
    DeltaMatMul,
    computeWeightGradientsForWeightsInConnectedLayer,
    computeBiasGradsForConnected_Layer,
    scaleGrad,
    SGD,
    Adam,
    ApplyPadding,
    Convolve,
    ConvolveDelta,
    DilateInput,
    Convolve,
    transConv,
    transConvBackward,
    computeBiasGradsForConv,
    computeKernelGradients,
    accumulateKernelGradsForTransConv,
    MaxPooling,
    MaxPoolDelta,
    element_wise_mul,
    scaleDiff,
    element_wise_sub,
    mse,
    mae,
    categorical_cross_entropy,
    sparse_categorical_cross_entropy,
    binary_cross_entropy,
    recurrentMatMul,
    recurrentTimeDelta,
    recurrentWeightGradsAccumulation,
    recurrentBiasGradsAccumulation,
    gradientClipping,
}
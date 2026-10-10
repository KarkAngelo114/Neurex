const { unpackQKVO, transpose2D, concatenateFloat32Array } = require("../../../utils/utils");

const Relu = (arr) => {
    const output = new Float32Array(arr.length);

    for (let i = 0; i < output.length; i++) {
        output[i] = Math.max(arr[i], 0);
    }
    return output;
};

const Sigmoid = (arr) => {
    const output = new Float32Array(arr.length);
    for (let i = 0; i < output.length; i++) {
        output[i] = 1 / (1 + Math.exp(-arr[i]));
    }
    return output;
};

const Tanh = (arr) => {
    const output = new Float32Array(arr.length);
    for (let i = 0; i < output.length; i++) {
        output[i] = Math.tanh(arr[i]);
    }
    return output;
};

const Softmax = (arr) => {
    const output = new Float32Array(arr.length);
    const maxVal = Math.max(...arr);
    let sum = 0;

    for (let i = 0; i < output.length; i++) {
        output[i] = Math.exp(arr[i] - maxVal);
        sum += output[i];
    }

    for (let i = 0; i < output.length; i++) {
        output[i] /= sum;
    }

    return output;
};

const Linear = (arr) => arr;


const DReLu = (arr) => {
    const output = new Float32Array(arr.length);
    for (let i = 0; i < output.length; i++) {
        output[i] = arr[i] > 0 ? 1 : 0;
    }
    return output;
};

const DSigmoid = (arr) => {
    const output = new Float32Array(arr.length);
    for (let i = 0; i < output.length; i++) {
        const s = 1 / (1 + Math.exp(-arr[i]));
        output[i] = s * (1 - s);
    }
    return output;
};

const DTanh = (arr) => {
    const output = new Float32Array(arr.length);
    for (let i = 0; i < output.length; i++) {
        const t = Math.tanh(arr[i]);
        output[i] = 1 - t * t;
    }
    return output;
};

const DSoftmax = (arr1, arr2) => {
    let output = new Float32Array(arr1.length);

    let dot_product = 0;
    for (let i = 0; i < arr1.length; i++) {
        dot_product += arr2[i] * arr1[i];
    }

    for (let i = 0; i < arr1.length; i++) {
        output[i] = arr1[i] * (arr2[i] - dot_product);
    }

    return output;
};

const DLinear = (arr) => {
    const output = new Float32Array(arr.length);
    output.fill(1);
    return output;
};

const getEmbeddings = (tokenVector, embeddingDim, lookup) => {

    const output = new Float32Array(tokenVector.length * embeddingDim);

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

const MatMul = (input, inputSize, outputSize, weights, biases) => {

    const output = new Float32Array(outputSize);

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
const accumulateWeightsAndBiasGradsForConnectedLayer = (activations, delta, weightGrads, biasGrads, weightShape) => {
    const [inputSize, outputSize] = weightShape;

    for (let i = 0; i < delta.length; i++) {
        biasGrads[i] += delta[i];
    }

    for (let i = 0; i < inputSize; i++) {
        const inputVal = activations[i];

        const rowStart = i * outputSize;
        const rowEnd = rowStart + outputSize;
        const gradRow = weightGrads.subarray(rowStart, rowEnd);

        for (let j = 0; j < outputSize; j++) {
            gradRow[j] += inputVal * delta[j];
        }
    }

    return {
        weightGrads: weightGrads,
        biasGrads: biasGrads
    }
}

const scale = (input, scalingValue) => {
    const output = input;

    for (let i = 0; i < input.length; i++) {
        output[i] /= scalingValue;
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

const RMSProp = (params, grads, sqAvg, lr, epsilon, decayRate) => {

    for (let i = 0; i < params.length; i++) {
        sqAvg[i] = decayRate * sqAvg[i] + (1 - decayRate) * (grads[i] * grads[i]);
        params[i] -= (lr / (Math.sqrt(sqAvg[i]) + epsilon)) * grads[i];
    }

    return {
        params: params,
        sqAvg: sqAvg
    }
}

const ConvolveForward = (input, inputShape, outputShape, kernelShape, weights, biases) => {
    const [f, kh, kw, d] = kernelShape;
    const [iH, iW, iD] = inputShape;
    const [oH, oW, oD] = outputShape;

    if (d !== iD) {
        throw new Error(`ConvolveForward: kernel input depth (${d}) != input depth (${iD})`);
    }
    if (f !== oD) {
        throw new Error(`ConvolveForward: number of filters (${f}) != output depth (${oD})`);
    }
    if (weights.length !== f * kh * kw * d) {
        throw new Error(`ConvolveForward: expected ${f * kh * kw * d} weights, received ${weights.length}`);
    }

    const output = new Float32Array(oH * oW * oD);

    for (let h = 0; h < oH; h++) {
        for (let w = 0; w < oW; w++) {
            for (let filter = 0; filter < f; filter++) {
                let sum = biases ? biases[filter] : 0;

                for (let kernelH = 0; kernelH < kh; kernelH++) {
                    const inputH = h + kernelH;
                    if (inputH >= iH) continue;

                    for (let kernelW = 0; kernelW < kw; kernelW++) {
                        const inputW = w + kernelW;
                        if (inputW >= iW) continue;

                        const inputBase = (inputH * iW + inputW) * iD;
                        const weightBase = ((filter * kh + kernelH) * kw + kernelW) * d;
                        const channelLimit = d - (d % 4);

                        for (let channel = 0; channel < channelLimit; channel += 4) {
                            sum += input[inputBase + channel] * weights[weightBase + channel];
                            sum += input[inputBase + channel + 1] * weights[weightBase + channel + 1];
                            sum += input[inputBase + channel + 2] * weights[weightBase + channel + 2];
                            sum += input[inputBase + channel + 3] * weights[weightBase + channel + 3];
                        }

                        for (let channel = channelLimit; channel < d; channel++) {
                            sum += input[inputBase + channel] * weights[weightBase + channel];
                        }
                    }
                }

                output[(h * oW + w) * oD + filter] = sum;
            }
        }
    }

    return output;
}

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

const ConvolveBackward = (input, OutputProjectionShape, deltaInputShape, kernelShape, kernels) => {
    // If convolve forward does the shrinking, this function does the opposite. The delta to project needs to be larger than what's coming in
    const [oH, oW, oD] = OutputProjectionShape; // target output shape of this function
    const [deltaH, deltaW, deltaD] = deltaInputShape; // current delta shape coming in
    const [f, kh, kw, d] = kernelShape;

    if (deltaD !== f) {
        throw new Error(`ConvolveBackward: delta depth (${deltaD}) != number of filters (${f})`);
    }
    if (oD !== d) {
        throw new Error(`ConvolveBackward: output depth (${oD}) != kernel input depth (${d})`);
    }
    if (input.length !== deltaH * deltaW * deltaD) {
        throw new Error(`ConvolveBackward: expected ${deltaH * deltaW * deltaD} input values, received ${input.length}`);
    }
    if (kernels.length !== f * kh * kw * d) {
        throw new Error(`ConvolveBackward: expected ${f * kh * kw * d} weights, received ${weights.length}`);
    }

    const weights = RotateKernels(f, kh, kw, d, kernels);
    const output = new Float32Array(oH * oW * oD);

    // Scatter each incoming delta through the weights used by the forward convolution.
    for (let h = 0; h < deltaH; h++) {
        for (let w = 0; w < deltaW; w++) {
            const deltaBase = (h * deltaW + w) * deltaD;

            for (let filter = 0; filter < f; filter++) {
                const deltaValue = input[deltaBase + filter];

                for (let kernelH = 0; kernelH < kh; kernelH++) {
                    const outputH = h + kernelH;
                    if (outputH >= oH) continue;

                    for (let kernelW = 0; kernelW < kw; kernelW++) {
                        const outputW = w + kernelW;
                        if (outputW >= oW) continue;

                        const outputBase = (outputH * oW + outputW) * oD;
                        const weightBase = ((filter * kh + kernelH) * kw + kernelW) * d;

                        for (let channel = 0; channel < d; channel++) {
                            output[outputBase + channel] += deltaValue * weights[weightBase + channel];
                        }
                    }
                }
            }
        }
    }

    return output;
}

const AccumulateWeightAndBiasGradsForConv = (activations_outputs, delta, weightGrads, biasGrads, inputShape, outputShape, kernelShape, stride) => {
    const [inputH, inputW, Cin] = inputShape;
    const [H, W, Cout] = outputShape;
    const [numFilters, Kh, Kw, d] = kernelShape;

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

                                sum0 += activations_outputs[baseInputIndex + c] * deltaVal;
                                sum1 += activations_outputs[baseInputIndex + c + 1] * deltaVal;
                                sum2 += activations_outputs[baseInputIndex + c + 2] * deltaVal;
                                sum3 += activations_outputs[baseInputIndex + c + 3] * deltaVal;
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
                                sum += activations_outputs[inputIndex] * delta[deltaIndex];
                            }
                        }
                    }

                    const gradIndex = kernelRowOffset * Cin + c;
                    weightGrads[gradIndex] += sum;
                }
            }
        }
    }

    for (let f = 0; f < numFilters; f++) {
        let sum = 0;

        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const idx = (h * H + w) * numFilters + f;
                sum += delta[idx];
            }
        }

        biasGrads[f] += sum;
    }

    return {
        weightGrads: weightGrads,
        biasGrads: biasGrads
    }

}

const AccumulateGammaAndBetaGrads = (gammaGrads, dGamma, betaGrads, dBeta) => {

    for (let i = 0; i < dGamma.length; i++) {
        gammaGrads[i] += dGamma[i];
        betaGrads[i] += dBeta[i];
    }

    return {
        gammaGrads: gammaGrads,
        betaGrads: betaGrads
    }
}

const MaxPooling = (arr, pool_size, inputShape, outputShape, strides) => {
    const [poolH, poolW] = pool_size;
    const [inputH, inputW, inputD] = inputShape;
    const [outputH, outputW, outputD] = outputShape;

    const output =  new Float32Array(outputH * outputW * outputD);
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

const recurrentMatMul = (input, prevHiddenState,  inputWeightShape, recurrentWeightShape, weights, biases) => {
    // The weights were concatenated during initialization as:
    // [input_weights..., recurrent_weights...]
    const inputSize = inputWeightShape[0];
    const units = inputWeightShape[1];
    const range_input_weights = inputSize * units;

    const output = new Float32Array(units);

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

const transConv = (input, inputShape, outputShape, strides, filters, weightShape, weights, biases) => {
    
    const [iH, iW, iD] = inputShape;
    const [oH, oW, oD] = outputShape;
    const [f, kh, kw, d] = weightShape;

    const output = new Float32Array(oH * oW * oD);

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

    const deltaOutputIndex = (y, x, f) => (y * oW + x) * oD + f;

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

const accumulateWeightandBiasGradsForTransConv = (activation_outputs, deltas, weightGrads, biasGrads, inputShape, outputShape, weightShape, strides,) => {
    const [iH, iW, iD] = inputShape;
    const [oH, oW, oD] = outputShape;
    const [filters, kh, kw, d] = weightShape;

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

    for (let f = 0; f < filters; f++) {
        let sum = 0;

        for (let h = 0; h < oH; h++) {
            for (let w = 0; w < oW; w++) {
                const idx = (h * oW + w) * filters + f;
                sum += deltas[idx];
            }
        }

        biasGrads[f] += sum;
    }

    return {
        weightGrads: weightGrads,
        biasGrads: biasGrads
    };
}

const dotProduct = (arr1, arr2, inputSize, outputSize) => {
    const output = new Float32Array(outputSize);

    for (let i = 0; i < inputSize; i++) {
        const inputVal = arr1[i];

        const rowStart = i * outputSize;
        const rowEnd = rowStart + outputSize;
        const inputVal2 = arr2.subarray(rowStart, rowEnd);

        for (let j = 0; j < outputSize; j++) {
            output[j] += inputVal * inputVal2[j];
        }
    }

    return output;
}

const computelayerNorm = (input, size, gamma, beta, eps) => {
    let mean = 0;

    for (let i = 0; i < size; i++) {
        mean += input[i];
    };

    mean /= size;

    let variance = 0;
    for (let i = 0; i < size; i++) {
        variance += (input[i] - mean) ** 2;
    }

    variance /= size;

    const std = Math.sqrt(variance + eps);
    const outputs = new Float32Array(size);

    for (let i = 0; i < size; i++) {
        const xHat = (input[i] - mean) / std;
        outputs[i] = gamma[i] * xHat + beta[i];
    }

    return outputs;
}

function computeLayerNormBackward(dY, X, gamma, size, eps) {
    const dX = new Float32Array(size);
    const dGamma = new Float32Array(size);
    const dBeta = new Float32Array(size);

    // 1. Recompute forward statistics (Mean & Variance)
    let mean = 0;
    for (let i = 0; i < size; i++) {
        mean += X[i];
    }

    mean /= size;

    let variance = 0;
    for (let i = 0; i < size; i++) {
        variance += (X[i] - mean) ** 2;
    }
    variance /= size;

    const stdInv = 1.0 / Math.sqrt(variance + eps);

    // 2. Compute normalized values (xHat) and intermediate parameter gradients
    const xHat = new Float32Array(size);
    let sumDy = 0;
    let sumDyXhat = 0;

    for (let i = 0; i < size; i++) {
        xHat[i] = (X[i] - mean) * stdInv;
        
        // Parameter Gradients
        dBeta[i] = dY[i];
        dGamma[i] = dY[i] * xHat[i];

        // Accumulate scalar sums for input gradient equation
        const dyGamma = dY[i] * gamma[i];sumDy += dyGamma;
        sumDyXhat += dyGamma * xHat[i];
    }

    // 3. Compute Input Gradient (dX) using closed-form formula
    const invSize = 1.0 / size;
    for (let i = 0; i < size; i++) {
        const dyGamma = dY[i] * gamma[i];
        dX[i] = stdInv * (dyGamma - (sumDy * invSize) - (xHat[i] * sumDyXhat * invSize));
    }

    return { dX, dGamma, dBeta };
}

const accumulate_element_wise_mul = (arr1, arr2, arr3) => {
    for (let i = 0; i < arr1.length; i++) {
        arr3[i] += arr2[i] * arr1[i];
    }
    return arr3;
}


const projectToQKV = (
    input, 
    Q_weights, 
    Q_bias,
    K_weights, 
    K_bias,
    V_weights,
    V_bias,
    embeddingDim,
    sequenceLen
) => {
    const Q = new Float32Array(sequenceLen * embeddingDim);
    const K = new Float32Array(sequenceLen * embeddingDim);
    const V = new Float32Array(sequenceLen * embeddingDim);

    for (let t = 0; t < sequenceLen; t++) {
        const tokenVec = input.subarray(t * embeddingDim, (t + 1) * embeddingDim);
        Q.set(MatMul(tokenVec, embeddingDim, embeddingDim, Q_weights, Q_bias), t * embeddingDim);
        K.set(MatMul(tokenVec, embeddingDim, embeddingDim, K_weights, K_bias), t * embeddingDim);
        V.set(MatMul(tokenVec, embeddingDim, embeddingDim, V_weights, V_bias), t * embeddingDim);
    }

    return {
        Q: Q,
        K: K,
        V: V
    }
}

const CoreAttention = (input, weights, biases, embedDim, seqLen, dkRoot) => {
    const {Q_weights, Q_bias, K_weights, K_bias, V_weights, V_bias} = unpackQKVO(weights, biases, null, null, embedDim);

    const {Q, K, V} = projectToQKV(input, Q_weights, Q_bias, K_weights, K_bias, V_weights, V_bias, embedDim, seqLen);

    const transpose_K = transpose2D(K, seqLen, embedDim);

    const scores = new Float32Array(seqLen * seqLen);
    for (let t = 0; t < seqLen; t++) {
        const Qrow = Q.subarray(t * embedDim, (t + 1) * embedDim);
        const rowScores = dotProduct(Qrow, transpose_K, embedDim, seqLen); // inputSize=embedDim, outputSize=seqLen
        scores.set(rowScores, t * seqLen);
    }

    const scaledvals = scale(scores, dkRoot);

    const softmaxOutput = new Float32Array(seqLen * seqLen);
    for (let t = 0; t < seqLen; t++) {
        const row = scaledvals.subarray(t * seqLen, (t + 1) * seqLen);
        softmaxOutput.set(Softmax(row), t * seqLen);
    }

    const output = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const srow = softmaxOutput.subarray(t * seqLen, (t + 1) * seqLen);
        const orow = dotProduct(srow, V, seqLen, embedDim); // inputSize=seqLen, outputSize=embedDim
        output.set(orow, t * embedDim);
    }

    const output_object = {
        X: input,
        Q: Q, 
        K: K, 
        V: V,
        S: softmaxOutput,
        output: output
    };

    return output_object;
}

const CoreAttentionBackward = (incomingDelta, Q, K, V, storedS, weights, embedDim, seqLen, dkRoot) => {
    // just like in feedforward, we unpack the weights, but we pass "null" to the 2nd - 4th argument of the function because we only want the weights for QKV
    const {Q_weights, K_weights, V_weights} = unpackQKVO(weights, null, null, null, embedDim);

    const transpose_V = transpose2D(V, seqLen, embedDim); 
    const dS = new Float32Array(seqLen * seqLen);
    for (let t = 0; t < seqLen; t++) {
        const deltaRow = incomingDelta.subarray(t * embedDim, (t + 1) * embedDim);
        dS.set(dotProduct(deltaRow, transpose_V, embedDim, seqLen), t * seqLen);
    }

    const transpose_S = transpose2D(storedS, seqLen, seqLen); // Sᵀ
    const dV = new Float32Array(seqLen * embedDim);
    for (let k = 0; k < seqLen; k++) {
        const sCol = transpose_S.subarray(k * seqLen, (k + 1) * seqLen);
        dV.set(dotProduct(sCol, incomingDelta, seqLen, embedDim), k * embedDim);
    }

    // apply the softmax derivative (Jacobian matrix)
    const dScaled = new Float32Array(seqLen * seqLen);
    for (let t = 0; t < seqLen; t++) {
        const sRow = storedS.subarray(t * seqLen, (t + 1) * seqLen);
        const dSRow = dS.subarray(t * seqLen, (t + 1) * seqLen);
        
        // we pass the sRow (storedS during feedfoward) and dSRow
        dScaled.set(DSoftmax(sRow, dSRow), t * seqLen);
    }

    const dScores = scale(dScaled, dkRoot);

    const dQ = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const dScoreRow = dScores.subarray(t * seqLen, (t + 1) * seqLen);
        dQ.set(dotProduct(dScoreRow, K, seqLen, embedDim), t * embedDim);
    }

    const transpose_dScores = transpose2D(dScores, seqLen, seqLen);
    const dK = new Float32Array(seqLen * embedDim);
    for (let k = 0; k < seqLen; k++) {
        const col = transpose_dScores.subarray(k * seqLen, (k + 1) * seqLen);
        dK.set(dotProduct(col, Q, seqLen, embedDim), k * embedDim);
    }

    const transpose_Qw = transpose2D(Q_weights, embedDim, embedDim);
    const transpose_Kw = transpose2D(K_weights, embedDim, embedDim);
    const transpose_Vw = transpose2D(V_weights, embedDim, embedDim);

    const dX = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const dQrow = dQ.subarray(t * embedDim, (t + 1) * embedDim);
        const dKrow = dK.subarray(t * embedDim, (t + 1) * embedDim);
        const dVrow = dV.subarray(t * embedDim, (t + 1) * embedDim);

        const fromQ = dotProduct(dQrow, transpose_Qw, embedDim, embedDim);
        const fromK = dotProduct(dKrow, transpose_Kw, embedDim, embedDim);
        const fromV = dotProduct(dVrow, transpose_Vw, embedDim, embedDim);

        for (let d = 0; d < embedDim; d++) {
            dX[t * embedDim + d] = fromQ[d] + fromK[d] + fromV[d];
        }
    }


    const data_object = {
        dQ: dQ,
        dK: dK,
        dV: dV,
        dX: dX
    }

    return data_object;
}

const CoreMultiHeadAttention = (input, weights, biases, embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking) => {
    // 1. Unpack Q, K, V, and O
    const { Q_weights, Q_bias, K_weights, K_bias, V_weights, V_bias, O_weights, O_bias } = unpackQKVO(weights, biases, null, null, embedDim, true);

    // 2. Project Input to Q, K, V [seqLen, embedDim]
    const {Q, K, V} = projectToQKV(input, Q_weights, Q_bias, K_weights, K_bias, V_weights, V_bias, embedDim, seqLen);

    const mhaOutput = new Float32Array(seqLen * embedDim);

    const scoresPerHead = seqLen * seqLen;
    const S_per_head = new Float32Array(numHeads * scoresPerHead);

    // 3. Process Each Head Independently
    for (let h = 0; h < numHeads; h++) {
        const headOffset = h * headDim;

        // Extract Head-specific Q, K, V slices [seqLen, headDim]
        const Q_h = new Float32Array(seqLen * headDim);
        const K_h = new Float32Array(seqLen * headDim);
        const V_h = new Float32Array(seqLen * headDim);

        for (let t = 0; t < seqLen; t++) {
            Q_h.set(Q.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            K_h.set(K.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            V_h.set(V.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
        }

        // just like in simple attention, we ran dot product each Q * (K^T)
        const transpose_K_h = transpose2D(K_h, seqLen, headDim);
        const scores = new Float32Array(seqLen * seqLen);
        for (let t = 0; t < seqLen; t++) {
            const Qrow = Q_h.subarray(t * headDim, (t + 1) * headDim);
            const rowScores = dotProduct(Qrow, transpose_K_h, headDim, seqLen);
            scores.set(rowScores, t * seqLen);
        }

        if (useCausalMasking) {
            for (let i = 0; i < seqLen; i++) {
                for (let j = i + 1; j < seqLen; j++) {
                    scores[i * seqLen + j] = -1e9;
                }
            }
        }
    
        // scale scores
        const scaledVals = scale(scores, dkRoot);

        // apply softmax to scaled scores
        const softmaxOutput = new Float32Array(seqLen * seqLen);
        for (let t = 0; t < seqLen; t++) {
            const row = scaledVals.subarray(t * seqLen, (t + 1) * seqLen);
            const softmaxRow = Softmax(row);
            softmaxOutput.set(softmaxRow, t * seqLen);
        }
        S_per_head.set(softmaxOutput, h * scoresPerHead);

        // Multiply Softmax Scores with V_h & Write Back to Concatenated Array
        for (let t = 0; t < seqLen; t++) {
            const srow = softmaxOutput.subarray(t * seqLen, (t + 1) * seqLen);
            const headOutRow = dotProduct(srow, V_h, seqLen, headDim);
            
            // Insert back into the target head position in mhaOutput
            const targetIdx = t * embedDim + headOffset;
            mhaOutput.set(headOutRow, targetIdx);
        }
    }

    // Step 4. Final Linear Projection (W_O) [seqLen, embedDim]
    let finalOutput = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const mhaRow = mhaOutput.subarray(t * embedDim, (t + 1) * embedDim);
        finalOutput.set(MatMul(mhaRow, embedDim, embedDim, O_weights, O_bias), t * embedDim);
    }

    const output_object = {
        X: input,
        Q: Q, 
        K: K, 
        V: V,
        mhaOutput: mhaOutput,
        S_perHead: S_per_head,
        finalOutput: finalOutput
    };

    return output_object;
}

const CoreMultiHeadAttentionBackward = (incomingDelta, weights, Q, K, V, S_perHead, embedDim, seqLen, numHeads, headDim, dkRoot, useCausalMasking) => {
    const {Q_weights, K_weights, V_weights, O_weights} = unpackQKVO(weights, null, null, null, embedDim, true);

    // first we get the dMHAoutput by projecting the incoming delta to transposed O_weights
    const transposed_O = transpose2D(O_weights, embedDim, embedDim);
    const dMhaOutput = new Float32Array(embedDim * seqLen);
    for (let i = 0; i < seqLen; i++) {
        const incomingDeltaRow = incomingDelta.subarray(i * embedDim, (i + 1) * embedDim);
        dMhaOutput.set(dotProduct(incomingDeltaRow, transposed_O, embedDim, embedDim), i * embedDim);
    }

    // Per head
    const dQ = new Float32Array(seqLen * embedDim);
    const dK = new Float32Array(seqLen * embedDim);
    const dV = new Float32Array(seqLen * embedDim);

    for (let h = 0; h < numHeads; h++) {
        const headOffset = h * headDim;

        const S = S_perHead.subarray(h * seqLen * seqLen, (h + 1) * seqLen * seqLen);

        // slice this head's Q_h, K_h, V_h, and its share of dMhaOutput
        const Q_h = new Float32Array(seqLen * headDim);
        const K_h = new Float32Array(seqLen * headDim);
        const V_h = new Float32Array(seqLen * headDim);
        const dHeadOut = new Float32Array(seqLen * headDim);
        for (let t = 0; t < seqLen; t++) {
            Q_h.set(Q.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            K_h.set(K.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            V_h.set(V.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
            dHeadOut.set(dMhaOutput.subarray(t * embedDim + headOffset, t * embedDim + headOffset + headDim), t * headDim);
        }

        const transpose_Vh = transpose2D(V_h, seqLen, headDim);
        const dS = new Float32Array(seqLen * seqLen);
        for (let t = 0; t < seqLen; t++) {
            dS.set(dotProduct(dHeadOut.subarray(t * headDim, (t + 1) * headDim), transpose_Vh, headDim, seqLen), t * seqLen);
        }

        const transpose_S = transpose2D(S, seqLen, seqLen);
        const dV_h = new Float32Array(seqLen * headDim);
        for (let k = 0; k < seqLen; k++) {
            dV_h.set(dotProduct(transpose_S.subarray(k * seqLen, (k + 1) * seqLen), dHeadOut, seqLen, headDim), k * headDim);
        }

        const dScaled = new Float32Array(seqLen * seqLen);
        for (let t = 0; t < seqLen; t++) {
            const sRow = S.subarray(t * seqLen, (t + 1) * seqLen);
            const dSRow = dS.subarray(t * seqLen, (t + 1) * seqLen);
            dScaled.set(DSoftmax(sRow, dSRow), t * seqLen);
        }

        const dScores = scale(dScaled, dkRoot);

        if (useCausalMasking) {
            for (let i = 0; i < seqLen; i++) {
                for (let j = i + 1; j < seqLen; j++) {
                    dScores[i * seqLen + j] = 0;
                }
            }
        }

        const dQ_h = new Float32Array(seqLen * headDim);
        for (let t = 0; t < seqLen; t++) {
            dQ_h.set(dotProduct(dScores.subarray(t * seqLen, (t + 1) * seqLen), K_h, seqLen, headDim), t * headDim);
        }
        
        const transpose_dScores = transpose2D(dScores, seqLen, seqLen);
        const dK_h = new Float32Array(seqLen * headDim);
        for (let k = 0; k < seqLen; k++) {
            dK_h.set(dotProduct(transpose_dScores.subarray(k * seqLen, (k + 1) * seqLen), Q_h, seqLen, headDim), k * headDim);
        }

        // write this head's contribution back into the FULL embedDim-wide buffers
        for (let t = 0; t < seqLen; t++) {
            dQ.set(dQ_h.subarray(t * headDim, (t + 1) * headDim), t * embedDim + headOffset);
            dK.set(dK_h.subarray(t * headDim, (t + 1) * headDim), t * embedDim + headOffset);
            dV.set(dV_h.subarray(t * headDim, (t + 1) * headDim), t * embedDim + headOffset);
        }
    }

    const transpose_Qw = transpose2D(Q_weights, embedDim, embedDim);
    const transpose_Kw = transpose2D(K_weights, embedDim, embedDim);
    const transpose_Vw = transpose2D(V_weights, embedDim, embedDim);
    const dX = new Float32Array(seqLen * embedDim);
    for (let t = 0; t < seqLen; t++) {
        const fromQ = dotProduct(dQ.subarray(t * embedDim, (t + 1) * embedDim), transpose_Qw, embedDim, embedDim);
        const fromK = dotProduct(dK.subarray(t * embedDim, (t + 1) * embedDim), transpose_Kw, embedDim, embedDim);
        const fromV = dotProduct(dV.subarray(t * embedDim, (t + 1) * embedDim), transpose_Vw, embedDim, embedDim);
        for (let d = 0; d < embedDim; d++) {
            dX[t * embedDim + d] = fromQ[d] + fromK[d] + fromV[d]
        };
    }


    const data = {
        dQ: dQ,
        dK: dK,
        dV: dV,
        dMhaOutput: dMhaOutput,
        dX: dX
    }

    return data;
}

const jaccard = (arr1, arr2) => {
    const set1 = new Set(arr1);
    const set2 = new Set(arr2);

    const intersection = [...set1].filter(x => set2.has(x));
    const union = new Set([...set1, ...set2]);

    // ensure returned values are not negative
    return Math.abs(union.size === 0 ? 0 : intersection.length / union.size);
}

const element_wise_add = (arr1, arr2) => {
    const output = new Float32Array(arr1.length);

    for (let i = 0; i < arr1.length; i++) {
        output[i] = arr1[i] + arr2[i];
    }

    return output;
}

const SinusoidalPositionalEncoding = (input, embeddingDim, sequenceLength) => {
   if (input.length !== embeddingDim * sequenceLength) {
        throw new Error(
            `Sinusoidal positional encoding shape mismatch: input length ${input.length}, ` +
            `expected ${embeddingDim * sequenceLength}`
        );
    }

    const output = new Float32Array(input);

    for (let pos = 0; pos < sequenceLength; pos++) {
        const offset = pos * embeddingDim;

        let i = 0;

        // Unroll the embedding-dimension loop four times per iteration.
        for (; i <= embeddingDim - 4; i += 4) {
            const pairIndex = Math.floor(i / 2);
            const exponent = (2 * pairIndex) / embeddingDim;
            const angle = pos / Math.pow(10000, exponent);

            output[offset + i] += (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);

            const pairIndex1 = Math.floor((i + 1) / 2);
            const exponent1 = (2 * pairIndex1) / embeddingDim;
            const angle1 = pos / Math.pow(10000, exponent1);
            output[offset + i + 1] += (i % 2 === 1) ? Math.sin(angle1) : Math.cos(angle1);

            const pairIndex2 = Math.floor((i + 2) / 2);
            const exponent2 = (2 * pairIndex2) / embeddingDim;
            const angle2 = pos / Math.pow(10000, exponent2);
            output[offset + i + 2] += (i % 2 === 0) ? Math.sin(angle2) : Math.cos(angle2);

            const pairIndex3 = Math.floor((i + 3) / 2);
            const exponent3 = (2 * pairIndex3) / embeddingDim;
            const angle3 = pos / Math.pow(10000, exponent3);
            output[offset + i + 3] += (i % 2 === 1) ? Math.sin(angle3) : Math.cos(angle3);
        }

        for (; i < embeddingDim; i++) {
            const pairIndex = Math.floor(i / 2);
            const exponent = (2 * pairIndex) / embeddingDim;
            const angle = pos / Math.pow(10000, exponent);

            output[offset + i] += (i % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
        }
    }

    return output;
}

const accumulateAttentionWeightsGradients = (dQ, dK, dV, dMhaOutput, mhaOutput, activation_outputs, weightGrads, embedDim, seqLen) => {
    const { Q_weightGrads: QwGrads, K_weightGrads: KwGrads, V_weightGrads: VwGrads, O_weightGrads: OwGrads } = unpackQKVO(null, null, weightGrads, null, embedDim, true);

    let QwG;
    let KwG;
    let VwG;
    let OwG;

    for (let t = 0; t < seqLen; t++) {
        const Xrow  = activation_outputs.subarray(t * embedDim, (t + 1) * embedDim);
        const dQrow = dQ.subarray(t * embedDim, (t + 1) * embedDim);
        const dKrow = dK.subarray(t * embedDim, (t + 1) * embedDim);
        const dVrow = dV.subarray(t * embedDim, (t + 1) * embedDim);

        QwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dQrow, QwGrads, embedDim, embedDim);
        KwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dKrow, KwGrads, embedDim, embedDim);
        VwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dVrow, VwGrads, embedDim, embedDim);

        const mhaRow = mhaOutput.subarray(t * embedDim, (t + 1) * embedDim);
        const dMhaRow = dMhaOutput.subarray(t * embedDim, (t + 1) * embedDim);
        OwG = computeWeightGradientsForWeightsInConnectedLayer(mhaRow, dMhaRow, OwGrads, embedDim, embedDim);
    }

    return concatenateFloat32Array([QwG, KwG, VwG, OwG]);
}

const accumulateAttentionBiasGrads = (dQ, dK, dV, dMhaOutput, biasGrads, embedDim, seqLen) => {
    const { Q_biasGrads: QbGrads, K_biasGrads: KbGrads, V_biasGrads: VbGrads, O_biasGrads: ObGrads }= unpackQKVO(null, null, null, biasGrads, embedDim, true);

    let QbG; 
    let KbG;
    let VbG;
    let ObG;

    for (let t = 0; t < seqLen; t++) {
        QbG = computeBiasGradsForConnected_Layer(QbGrads, dQ.subarray(t * embedDim, (t + 1) * embedDim));
        KbG = computeBiasGradsForConnected_Layer(KbGrads, dK.subarray(t * embedDim, (t + 1) * embedDim));
        VbG = computeBiasGradsForConnected_Layer(VbGrads, dV.subarray(t * embedDim, (t + 1) * embedDim));
        ObG = computeBiasGradsForConnected_Layer(ObGrads, dMhaOutput.subarray(t * embedDim, (t + 1) * embedDim));
    }
    
    return concatenateFloat32Array([QbG, KbG, VbG, ObG]);
}


const accumulateSimpleAttentionWeightGrads = (dQ, dK, dV, activation_outputs, weightGrads, embedDim, seqLen) => {
    const { Q_weightGrads: QwGrads, K_weightGrads: KwGrads, V_weightGrads: VwGrads } = unpackQKVO(null, null, weightGrads, null, embedDim);

    let QwG;
    let KwG;
    let VwG;

    for (let t = 0; t < seqLen; t++) {
        const Xrow  = activation_outputs.subarray(t * embedDim, (t + 1) * embedDim);
        const dQrow = dQ.subarray(t * embedDim, (t + 1) * embedDim);
        const dKrow = dK.subarray(t * embedDim, (t + 1) * embedDim);
        const dVrow = dV.subarray(t * embedDim, (t + 1) * embedDim);

        QwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dQrow, QwGrads, embedDim, embedDim);
        KwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dKrow, KwGrads, embedDim, embedDim);
        VwG = computeWeightGradientsForWeightsInConnectedLayer(Xrow, dVrow, VwGrads, embedDim, embedDim);
    }

    return concatenateFloat32Array([QwG, KwG, VwG]);
}

const accumulateSimpleAttentionBiasGrads = (dQ, dK, dV, biasGrads, embedDim, seqLen) => {
    const { Q_biasGrads: QbGrads, K_biasGrads: KbGrads, V_biasGrads: VbGrads } = unpackQKVO(null, null, null, biasGrads, embedDim);
    
    let QbG;
    let KbG;
    let VbG;

    for (let t = 0; t < seqLen; t++) {
        QbG = computeBiasGradsForConnected_Layer(QbGrads, dQ.subarray(t * embedDim, (t + 1) * embedDim));
        KbG = computeBiasGradsForConnected_Layer(KbGrads, dK.subarray(t * embedDim, (t + 1) * embedDim));
        VbG = computeBiasGradsForConnected_Layer(VbGrads, dV.subarray(t * embedDim, (t + 1) * embedDim));
    }
    
    return concatenateFloat32Array([QbG, KbG, VbG]);
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
    accumulateWeightsAndBiasGradsForConnectedLayer,
    scale,
    SGD,
    Adam,
    RMSProp,
    ConvolveForward,
    ConvolveBackward,
    AccumulateWeightAndBiasGradsForConv,
    transConv,
    transConvBackward,
    accumulateWeightandBiasGradsForTransConv,
    MaxPooling,
    MaxPoolDelta,
    element_wise_mul,
    accumulate_element_wise_mul,
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
    computelayerNorm,
    jaccard,
    CoreAttention,
    CoreAttentionBackward,
    CoreMultiHeadAttention,
    CoreMultiHeadAttentionBackward,
    accumulateAttentionWeightsGradients,
    accumulateAttentionBiasGrads,
    element_wise_add,
    SinusoidalPositionalEncoding,
    accumulateSimpleAttentionWeightGrads,
    accumulateSimpleAttentionBiasGrads,
    computeLayerNormBackward,
    AccumulateGammaAndBetaGrads,
}
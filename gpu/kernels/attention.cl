// helper functions (unpacking QKV weights and biases, optimized MatMul Helper)

static void _helper_matMul(
    __global const float* input,
    int inputSize,
    int outputSize,
    __global const float* weights,
    __global const float* biases,
    __global float* output
) {

    int j = 0;
    for (; j <= outputSize - 4; j += 4) {
        output[j + 0] = biases[j + 0];
        output[j + 1] = biases[j + 1];
        output[j + 2] = biases[j + 2];
        output[j + 3] = biases[j + 3];
    }


    for (; j < outputSize; j++) {
        output[j] = biases[j];
    }

    for (int i = 0; i < inputSize; i++) {
        float inputVal = input[i];
        int rowStart = i * outputSize;
        int base = 0;

        for (; base <= outputSize - 4; base += 4) {
            output[base + 0] += inputVal * weights[rowStart + base + 0];
            output[base + 1] += inputVal * weights[rowStart + base + 1];
            output[base + 2] += inputVal * weights[rowStart + base + 2];
            output[base + 3] += inputVal * weights[rowStart + base + 3];
        }

        // Cleanup loop for matrix elements
        for (; base < outputSize; ++base) {
            output[base] += inputVal * weights[rowStart + base];
        }
    }
}

__kernel void projectQKV(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* Q,
    __global float* K,
    __global float* V,
    const int embeddingDim,
    const int seqLen
) {
    int t = get_global_id(0);

    if (t >= seqLen) return;

    int offset = t * embeddingDim;
    int matrixSize = embeddingDim * embeddingDim;

    __global const float* tokenVec = input + offset;

    __global const float* Q_w = weights;
    __global const float* K_w = weights + matrixSize;
    __global const float* V_w = weights + matrixSize * 2;

    __global const float* Q_b = biases;
    __global const float* K_b = biases + embeddingDim;
    __global const float* V_b = biases + embeddingDim * 2;

    _helper_matMul(tokenVec, embeddingDim, embeddingDim, Q_w, Q_b, Q + offset);
    _helper_matMul(tokenVec, embeddingDim, embeddingDim, K_w, K_b, K + offset);
    _helper_matMul(tokenVec, embeddingDim, embeddingDim, V_w, V_b, V + offset);
}
__kernel void matmul(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output_tensor_template,
    const int inputSize,
    const int outputSize)
{
    int j = get_global_id(0);
    if (j >= outputSize) return;

    float acc = biases[j];
    int i = 0;

    // Main loop unrolled by 4
    for (; i <= inputSize - 4; i += 4) {
        acc += input[i]     * weights[i * outputSize + j];
        acc += input[i + 1] * weights[(i + 1) * outputSize + j];
        acc += input[i + 2] * weights[(i + 2) * outputSize + j];
        acc += input[i + 3] * weights[(i + 3) * outputSize + j];
    }

    // Cleanup loop for any remaining elements (0 to 3 elements)
    for (; i < inputSize; ++i) {
        acc += input[i] * weights[i * outputSize + j];
    }

    output_tensor_template[j] = acc;
}
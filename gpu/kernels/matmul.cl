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
    for (int i = 0; i < inputSize; ++i) {
        acc += input[i] * weights[i * outputSize + j];
    }
    output_tensor_template[j] = acc;
}
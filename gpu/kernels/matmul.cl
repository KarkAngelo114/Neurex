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


__kernel void dot_product(
    __global const float* arr1,
    __global const float* arr2,
    __global float* output,
    const int inputSize,
    const int outputSize
) {
    int j = get_global_id(0);

    if (j >= outputSize) return;

    float sum = 0.0f;

    for (int i = 0; i < inputSize; ++i) {
        sum += arr1[i] * arr2[i * outputSize + j];
    }

    output[j] = sum;
}
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

    for (; i + 3 < inputSize; i += 4) {
        acc += input[i] * weights[i * outputSize + j];
        acc += input[i + 1] * weights[(i + 1) * outputSize + j];
        acc += input[i + 2] * weights[(i + 2) * outputSize + j];
        acc += input[i + 3] * weights[(i + 3) * outputSize + j];
    }

    for (; i < inputSize; ++i) {
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
    
    int i = 0;

    for (; i + 3 < inputSize; i += 4) {
        sum += arr1[i] * arr2[i * outputSize + j];
        sum += arr1[i + 1] * arr2[(i + 1) * outputSize + j];
        sum += arr1[i + 2] * arr2[(i + 2) * outputSize + j];
        sum += arr1[i + 3] * arr2[(i + 3) * outputSize + j];
    }


    for (; i < inputSize; ++i) {
        sum += arr1[i] * arr2[i * outputSize + j];
    }

    output[j] = sum;
}
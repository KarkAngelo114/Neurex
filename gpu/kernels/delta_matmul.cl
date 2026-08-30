__kernel void delta_matmul(
    __global const float* delta,
    __global const float* weights,
    __global float* output,
    const int inputSize,
    const int outputSize)
{
    int i = get_global_id(0);
    if (i >= inputSize) return;

    float sum = 0.0f;

    int j = 0;

    int base = i * outputSize;

    for (int j = 0; j + 3 < outputSize; j += 4) {
        sum += weights[base + j] * delta[j];
        sum += weights[base + j + 1] * delta[j + 1];
        sum += weights[base + j + 2] * delta[j + 2];
        sum += weights[base + j + 3] * delta[j + 3];
    }

    for (; j < outputSize; ++j) {
        sum += weights[base + j] * delta[j];
    }

    output[i] = sum;
}
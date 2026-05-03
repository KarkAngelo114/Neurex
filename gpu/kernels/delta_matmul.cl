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
    int offset = i * outputSize;
    for (int j = 0; j < outputSize; ++j) {
        sum += weights[offset + j] * delta[j];
    }
    output[i] = sum;
}
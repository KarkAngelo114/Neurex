__kernel void recurrentMatMul(
    __global const float* input_data,
    __global const float* prevHiddenState,
    __global const float* input_weights,
    __global const float* recurrent_weights,
    __global const float* biases,
    __global float* output,
    const int inputSize,
    const int units
) {
    int j = get_global_id(0);
    if (j >= units) return;

    float z = biases[j];

    for (int i = 0; i < inputSize; ++i) {
        z += input_data[i] * input_weights[i * units + j];
    }

    for (int h = 0; h < units; ++h) {
        z += prevHiddenState[h] * recurrent_weights[h * units + j];
    }

    output[j] = z;
}

__kernel void recurrentTimeDelta(
    __global const float* delta,
    __global const float* recurrent_weights,
    __global float* output,
    const int C,
    const int D
) {
    int c = get_global_id(0);
    if (c >= C) return;

    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        sum += recurrent_weights[c * D + d] * delta[d];
    }

    output[c] = sum;
}
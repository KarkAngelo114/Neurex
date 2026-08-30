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

    int i = 0;
    for (; i + 3 < inputSize; i += 4) {
        z += input_data[i] * input_weights[i * units + j];
        z += input_data[i + 1] * input_weights[(i + 1) * units + j];
        z += input_data[i + 2] * input_weights[(i + 2) * units + j];
        z += input_data[i + 3] * input_weights[(i + 3) * units + j];
    }
    for (; i < inputSize; ++i) {
        z += input_data[i] * input_weights[i * units + j];
    }

    int h = 0;
    for (; h + 3 < units; h += 4) {
        z += prevHiddenState[h] * recurrent_weights[h * units + j];
        z += prevHiddenState[h + 1] * recurrent_weights[(h + 1) * units + j];
        z += prevHiddenState[h + 2] * recurrent_weights[(h + 2) * units + j];
        z += prevHiddenState[h + 3] * recurrent_weights[(h + 3) * units + j];
    }
    for (; h < units; ++h) {
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
    int d = 0;
    for (; d + 3 < D; d += 4) {
        sum += recurrent_weights[c * D + d] * delta[d];
        sum += recurrent_weights[c * D + d + 1] * delta[d + 1];
        sum += recurrent_weights[c * D + d + 2] * delta[d + 2];
        sum += recurrent_weights[c * D + d + 3] * delta[d + 3];
    }
    for (; d < D; ++d) {
        sum += recurrent_weights[c * D + d] * delta[d];
    }

    output[c] = sum;
}
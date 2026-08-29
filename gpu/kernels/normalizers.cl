
inline float accumulate(__global const float* grads, int size) {
    float sum = 0.0f;

    for (int i = 0; i < size; ++i) {
        sum += grads[i] * grads[i];
    }

    return sum;
}


__kernel void gradientClipping(
    __global float* grads,
    const float threshold,
    const float size
) {

    int i = get_global_id(0);

    if (i >= size) return;

    float norm = sqrt(accumulate(grads, size));

    if (norm > threshold) {
        float scaleVal = threshold / norm;

        for (int i = 0; i < size; i++) {
            grads[i] *= scaleVal;
        }
    }
}

__kernel void layer_norm_standard_size(
    __global const float* input,
    __global const float* gamma,
    __global const float* beta,
    __global float* output,
    const float mean,
    const float standardizationValue,
    const int size
) {

    int i = get_global_id(0);

    if (i >= size) return;

    float xHat = (input[i] - mean) / standardizationValue;

    output[i] = gamma[i] * xHat + beta[i];

}
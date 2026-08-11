
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
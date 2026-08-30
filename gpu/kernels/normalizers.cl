__kernel void gradientClipping(
    __global float* grads,
    const float scalingVal,
    const int size
) {

    int i = get_global_id(0);

    if (i >= size) return;

    grads[i] *= scalingVal;

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
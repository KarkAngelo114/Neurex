__kernel void mse(
    __global const float* preds,
    __global const float* actuals,
    __global float* output,
    const int element_length
) {

    int i = get_global_id(0);

    if (i >= element_length) return;

    float difference = preds[i] - actuals[i];
    output[i] = difference * difference;
}

__kernel void mae(
    __global const float* preds,
    __global const float* actuals,
    __global float* output,
    const int element_length
) {
    int i = get_global_id(0);

    if (i >= element_length) return;

    output[i] = fabs(preds[i] - actuals[i]);
}

__kernel void cce(
    __global const float* preds,
    __global const float* actuals,
    __global float* output,
    const int element_length,
    const float epsilon
) {
    int i = get_global_id(0);

    if (i >= element_length) return;

    output[i] = actuals[i] * log(fmax(preds[i], epsilon));
}

__kernel void bce(
    __global const float* preds,
    __global const float* actuals,
    __global float* output,
    const int element_length,
    const float epsilon
) {
    int i = get_global_id(0);

    if (i >= element_length) return;

    float p = fmax(min(preds[i], 1 - epsilon), epsilon);
    output[i] = actuals[i] * log(p) + (1 - actuals[i]) * log(1 - p);
}
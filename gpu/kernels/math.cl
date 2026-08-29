__kernel void element_wise_mul(
    __global const float* arr1,
    __global const float* arr2,
    __global float* output,
    const int size
) {
    int i = get_global_id(0);

    if (i >= size) return;

    output[i] = arr1[i] * arr2[i];
}

__kernel void element_wise_sub(
    __global const float* arr1,
    __global const float* arr2,
    __global float* output,
    const int size
) {

    int i = get_global_id(0);

    if (i >= size) return;

    output[i] = arr1[i] - arr2[i];

}

__kernel void scale_diff(
    __global const float* arr1,
    __global const float* arr2,
    __global const float* arr3,
    __global float* output,
    const int size
) {
    int i = get_global_id(0);

    if (i >= size) return;

    float scale = 2.0f / size;

    output[i] = (arr1[i] - arr2[i]) * arr3[i] * scale;
}


__kernel void scale(
    __global float* input,
    const int scalingFactor,
    const int size
) {

    int i = get_global_id(0);
    
    if (size > 0) {
        input[i] = input[i] / (float)scalingFactor;
    }

}

__kernel void accumulate_element_wise_mul(
    __global const float* arr1,
    __global const float* arr2,
    __global float* arr3,
    const int size
) {
    int i = get_global_id(0);

    if (i >= size) return;

    arr3[i] += arr2[i] * arr1[i];
}
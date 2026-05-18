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

    output[i] = (arr1[i] - arr2[i]) * arr3[i];
}
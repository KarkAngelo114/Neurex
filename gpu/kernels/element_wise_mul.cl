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
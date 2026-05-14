__kernel void computeBiasGradsForConnected_Layer(
    __global const float* delta,
    __global float* biasgrad,
    const int gradSize
) {
    int idx = get_global_id(0);
    if (idx < gradSize) {
        biasgrad[idx] += delta[idx];
    }
}
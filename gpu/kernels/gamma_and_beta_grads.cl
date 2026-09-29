
// shared kernel function for gradient accumulation of gamma and beta grads since the operation is identical
__kernel void accumulate_gamma_beta_grads(
    __global const float* deltas,
    __global float* grads,
    const int size
) {
    int i = get_global_id(0);

    if (i >= size) return;

    grads[i] += deltas[i];
}
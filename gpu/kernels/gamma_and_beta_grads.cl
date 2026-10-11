
// shared kernel function for gradient accumulation of gamma and beta grads since the operation is identical
__kernel void accumulate_gamma_beta_grads(
    __global const float* dGamma,
    __global const float* dBeta,
    __global float* gammaGrads,
    __global float* betaGrads,
    const int size
) {
    int i = get_global_id(0);

    if (i >= size) return;

    gammaGrads[i] += dGamma[i];
    betaGrads[i] += dBeta[i];
}
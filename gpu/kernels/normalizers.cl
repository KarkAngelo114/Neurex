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

inline void reduceLocal(
    __local float* values,
    uint localId,
    uint localSize
) {
    for (uint stride = localSize >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (localId < stride) {
            values[localId] += values[localId + stride];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
 * This implementation handles one layer-normalized vector.
 * It uses one workgroup and lets each work item process multiple
 * elements through a strided loop.
 */
__kernel void layer_norm_backward_one(
    __global const float* x,
    __global const float* dy,
    __global const float* gamma,

    __global float* dx,
    __global float* dgamma,
    __global float* dbeta,

    const int size,
    const float eps,

    __local float* scratch
) {
    const uint lid = get_local_id(0);
    const uint localSize = get_local_size(0);

    float localSum = 0.0f;

    for (int i = (int)lid; i < size; i += (int)localSize) {
        localSum += x[i];
    }

    scratch[lid] = localSum;
    reduceLocal(scratch, lid, localSize);

    const float mean = scratch[0] / (float)size;

    barrier(CLK_LOCAL_MEM_FENCE);

    float localVariance = 0.0f;

    for (int i = (int)lid; i < size; i += (int)localSize) {
        const float diff = x[i] - mean;
        localVariance += diff * diff;
    }

    scratch[lid] = localVariance;
    reduceLocal(scratch, lid, localSize);

    const float variance = scratch[0] / (float)size;
    const float invStd = rsqrt(variance + eps);

    barrier(CLK_LOCAL_MEM_FENCE);

    float localMeanG = 0.0f;

    for (int i = (int)lid; i < size; i += (int)localSize) {
        const float xHat = (x[i] - mean) * invStd;
        localMeanG += dy[i] * gamma[i];
    }

    scratch[lid] = localMeanG;
    reduceLocal(scratch, lid, localSize);

    const float meanG = scratch[0] / (float)size;

    barrier(CLK_LOCAL_MEM_FENCE);

    float localMeanGXHat = 0.0f;

    for (int i = (int)lid; i < size; i += (int)localSize) {
        const float xHat = (x[i] - mean) * invStd;
        const float g = dy[i] * gamma[i];

        localMeanGXHat += g * xHat;
    }

    scratch[lid] = localMeanGXHat;
    reduceLocal(scratch, lid, localSize);

    const float meanGXHat = scratch[0] / (float)size;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = (int)lid; i < size; i += (int)localSize) {
        const float xHat = (x[i] - mean) * invStd;
        const float g = dy[i] * gamma[i];

        dx[i] = invStd * (
            g - meanG - xHat * meanGXHat
        );

        dgamma[i] = dy[i] * xHat;
        dbeta[i] = dy[i];
    }
}
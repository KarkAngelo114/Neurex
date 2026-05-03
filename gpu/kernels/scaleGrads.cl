__kernel void scaleGrads(
    __global float* grads,
    const int batchSize
) {

    int i = get_global_id(0);
    
    if (batchSize > 0) {
        grads[i] = grads[i] / (float)batchSize;
    }

}
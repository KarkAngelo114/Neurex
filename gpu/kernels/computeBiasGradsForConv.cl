__kernel void computeBiasGradsForConv(
    __global float* bg,
    __global const float* d,
    const int oH,
    const int oW,
    const int F 
) {
    // Each work item processes one filter
    int f = get_global_id(0);
    
    if (f >= F) return;
    
    float sum = 0.0f;
    
    // Sum all deltas for this filter across spatial dimensions
    for (int h = 0; h < oH; h++) {
        for (int w = 0; w < oW; w++) {
            size_t idx = (h * oW + w) * F + f;
            sum += d[idx];
        }
    }
    
    bg[f] += sum;
}
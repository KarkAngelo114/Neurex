__kernel void convolve(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    const int strides,
    const int outputH,
    const int outputW,
    const int numFilters,
    const int kernelH,
    const int kernelW,
    const int depth,
    const int inputH,
    const int inputW
) {
    int y = get_global_id(0);
    int x = get_global_id(1);
    int f = get_global_id(2);

    // Bounds check
    if (y >= outputH || x >= outputW || f >= numFilters) return;

    int kernelSize = kernelH * kernelW * depth;

    // Equivalent to CPU:
    int baseY = y * strides;
    int baseX = x * strides;
    int outIndex = (y * outputW + x) * numFilters + f;

    float sum = biases[f];

    // Filter offset
    int filterOffset = f * kernelSize;

    for (int ky = 0; ky < kernelH; ky++) {
        int inY = baseY + ky;

        if (inY >= inputH) continue;

        for (int kx = 0; kx < kernelW; kx++) {
            int inX = baseX + kx;

            if (inX >= inputW) continue;

            int inputBase = (inY * inputW + inX) * depth;
            int kernelBase = filterOffset + (ky * kernelW + kx) * depth;

            int c = 0;
            
            for (; c <= depth - 4; c += 4) {
                sum += input[inputBase + c]     * weights[kernelBase + c];
                sum += input[inputBase + c + 1] * weights[kernelBase + c + 1];
                sum += input[inputBase + c + 2] * weights[kernelBase + c + 2];
                sum += input[inputBase + c + 3] * weights[kernelBase + c + 3];
            }

            // Remaining channels
            for (; c < depth; c++) {
                sum += input[inputBase + c] * weights[kernelBase + c];
            }
        }
    }

    output[outIndex] = sum;
}
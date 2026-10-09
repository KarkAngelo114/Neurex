__kernel void convolve(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    const int outputH,
    const int outputW,
    const int numFilters,
    const int kernelH,
    const int kernelW,
    const int depth,
    const int inputH,
    const int inputW
) {
    const int y = get_global_id(0);
    const int x = get_global_id(1);
    const int filter = get_global_id(2);

    if (y >= outputH || x >= outputW || filter >= numFilters) return;

    const int kernelSize = kernelH * kernelW * depth;
    const int outputIndex = (y * outputW + x) * numFilters + filter;
    const int filterOffset = filter * kernelSize;
    float sum = biases[filter];

    for (int kh = 0; kh < kernelH; kh++) {
        const int inputY = y + kh;
        if (inputY >= inputH) continue;

        for (int kw = 0; kw < kernelW; kw++) {
            const int inputX = x + kw;
            if (inputX >= inputW) continue;

            const int inputBase = (inputY * inputW + inputX) * depth;
            const int weightBase = filterOffset + (kh * kernelW + kw) * depth;
            int channel = 0;

            for (; channel <= depth - 4; channel += 4) {
                const float input0 = input[inputBase + channel];
                const float input1 = input[inputBase + channel + 1];
                const float input2 = input[inputBase + channel + 2];
                const float input3 = input[inputBase + channel + 3];
                const float weight0 = weights[weightBase + channel];
                const float weight1 = weights[weightBase + channel + 1];
                const float weight2 = weights[weightBase + channel + 2];
                const float weight3 = weights[weightBase + channel + 3];
                sum += input0 * weight0;
                sum += input1 * weight1;
                sum += input2 * weight2;
                sum += input3 * weight3;
            }

            for (; channel < depth; channel++) {
                sum += input[inputBase + channel] * weights[weightBase + channel];
            }
        }
    }

    output[outputIndex] = sum;
}

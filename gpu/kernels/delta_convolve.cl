__kernel void delta_convolve(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    const int deltaH,
    const int deltaW,
    const int numFilters,
    const int kernelH,
    const int kernelW,
    const int depth,
    const int outputH,
    const int outputW
) {
    const int h = get_global_id(0);
    const int w = get_global_id(1);
    const int channel = get_global_id(2);

    if (h >= outputH || w >= outputW || channel >= depth) return;

    float sum = 0.0f;
    for (int kh = 0; kh < kernelH; kh++) {
        const int deltaY = h - kh;
        if (deltaY < 0 || deltaY >= deltaH) continue;

        for (int kw = 0; kw < kernelW; kw++) {
            const int deltaX = w - kw;
            if (deltaX < 0 || deltaX >= deltaW) continue;

            const int deltaBase = (deltaY * deltaW + deltaX) * numFilters;
            const int kernelBase = ((kernelH - 1 - kh) * kernelW + (kernelW - 1 - kw)) * depth + channel;
            int filter = 0;

            for (; filter <= numFilters - 4; filter += 4) {
                const float delta0 = input[deltaBase + filter];
                const float delta1 = input[deltaBase + filter + 1];
                const float delta2 = input[deltaBase + filter + 2];
                const float delta3 = input[deltaBase + filter + 3];
                const int weight0 = filter * kernelH * kernelW * depth + kernelBase;
                const int weight1 = (filter + 1) * kernelH * kernelW * depth + kernelBase;
                const int weight2 = (filter + 2) * kernelH * kernelW * depth + kernelBase;
                const int weight3 = (filter + 3) * kernelH * kernelW * depth + kernelBase;
                sum += delta0 * weights[weight0];
                sum += delta1 * weights[weight1];
                sum += delta2 * weights[weight2];
                sum += delta3 * weights[weight3];
            }

            for (; filter < numFilters; filter++) {
                const int weight = filter * kernelH * kernelW * depth + kernelBase;
                sum += input[deltaBase + filter] * weights[weight];
            }
        }
    }

    output[(h * outputW + w) * depth + channel] = sum;
}

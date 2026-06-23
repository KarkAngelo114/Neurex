__kernel void delta_convolve(
    __global const float* input,
    __global const float* weights,
    __global float* output_tensor,
    const int Wp,
    const int C_in,
    const int F,
    const int KH,
    const int KW,
    const int C_k,
    const int oH,
    const int oW,
    const int stride
) {
    int h = get_global_id(0);
    int w = get_global_id(1);
    int c_out = get_global_id(2);

    if (h >= oH || w >= oW || c_out >= C_k) return;

    float sum = 0.0f;

    // ===== convolution accumulation with loop unrolling =====
    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            int ph = h * stride + kh;
            int pw = w * stride + kw;
            int baseInputIdx = (ph * Wp + pw) * C_in;
            int baseKernelIdx = ((kh) * KW + kw) * C_k;

            // Unrolled loop: process 4 filters at a time
            int f = 0;
            for (; f <= F - 4; f += 4) {
                int inputIdx0 = baseInputIdx + f;
                int inputIdx1 = baseInputIdx + f + 1;
                int inputIdx2 = baseInputIdx + f + 2;
                int inputIdx3 = baseInputIdx + f + 3;

                int kernelIdx0 = (f * KH + kh) * KW * C_k + baseKernelIdx + c_out;
                int kernelIdx1 = ((f + 1) * KH + kh) * KW * C_k + baseKernelIdx + c_out;
                int kernelIdx2 = ((f + 2) * KH + kh) * KW * C_k + baseKernelIdx + c_out;
                int kernelIdx3 = ((f + 3) * KH + kh) * KW * C_k + baseKernelIdx + c_out;

                sum += input[inputIdx0] * weights[kernelIdx0];
                sum += input[inputIdx1] * weights[kernelIdx1];
                sum += input[inputIdx2] * weights[kernelIdx2];
                sum += input[inputIdx3] * weights[kernelIdx3];
            }

            // Handle remaining filters
            for (; f < F; f++) {
                int inputIdx = baseInputIdx + f;
                int kernelIdx = (f * KH + kh) * KW * C_k + baseKernelIdx + c_out;
                sum += input[inputIdx] * weights[kernelIdx];
            }
        }
    }

    output_tensor[(h * oW + w) * C_k + c_out] = sum;
}
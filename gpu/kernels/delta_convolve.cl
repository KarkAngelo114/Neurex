__kernel void delta_convolve(
    __global const float* padded,
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

    // ===== convolution accumulation =====
    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            for (int f = 0; f < F; f++) {

                int ph = h * stride + kh;
                int pw = w * stride + kw;

                int inputIdx  = (ph * Wp + pw) * C_in + f;
                int kernelIdx = ((f * KH + kh) * KW + kw) * C_k + c_out;

                sum += padded[inputIdx] * weights[kernelIdx];
            }
        }
    }

    output_tensor[(h * oW + w) * C_k + c_out] = sum;
}
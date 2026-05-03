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
    const int oW
) {
    int gid = get_global_id(0);
    if (gid >= oH * oW * C_k) return;

    int c_out = gid % C_k;
    int hw = gid / C_k;
    int w = hw  % oW;
    int h = hw  / oW;

    float sum = 0.0f;

    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            for (int f = 0; f < F; f++) {
                int ph = h + kh;
                int pw = w + kw;

                int inputIdx  = (ph * Wp + pw) * C_in + f;
                int kernelIdx = ((f * KH + kh) * KW + kw) * C_k + c_out;

                sum += padded[inputIdx] * weights[kernelIdx];
            }
        }
    }

    output_tensor[(h * oW + w) * C_k + c_out] = sum;
}

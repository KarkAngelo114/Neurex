__kernel void computeKernelGradients(
    __global const float* input_Data,
    __global const float* d,
    __global float* wg,
    const int inputH,
    const int inputW,
    const int Cin,
    const int H,
    const int W,
    const int Cout,
    const int Kh,
    const int Kw,
    const int padH,
    const int padW
) {

    int f  = get_global_id(0);
    int kh = get_global_id(1);
    int z = get_global_id(2);

    int kw = z / Cin;
    int c  = z % Cin;

    if (f >= Cout || kh >= Kh || kw >= Kw || c >= Cin) {
        return;
    }

    float sum = 0.0f;

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {

            int inH = h + kh - padH;
            int inW = w + kw - padW;

            if (inH >= 0 && inH < inputH && inW >= 0 && inW < inputW) {

                size_t inputIndex = (inH * inputW + inW) * Cin + c;

                size_t deltaIndex = (h * W + w) * Cout + f;

                sum += input_Data[inputIndex] * d[deltaIndex];
            }
        }
    }

    size_t gradIndex = ((f * Kh + kh) * Kw + kw) * Cin + c;

    wg[gradIndex] += sum;
}
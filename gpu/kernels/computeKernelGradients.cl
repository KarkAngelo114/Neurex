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
    const int padW,
    const int stride
) {

    int f  = get_global_id(0);
    int kh = get_global_id(1);
    int z = get_global_id(2);

    int kw = z / (Cin / 4 + (Cin % 4 > 0 ? 1 : 0));
    int c_block = z % (Cin / 4 + (Cin % 4 > 0 ? 1 : 0));

    if (f >= Cout || kh >= Kh || kw >= Kw) {
        return;
    }

    int kernelRowOffset = (f * Kh + kh) * Kw + kw;
    
    // Process 4 channels at a time
    int c_start = c_block * 4;
    if (c_start >= Cin) {
        return;
    }
    
    int c_end = min(c_start + 4, Cin);
    
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {

            int inH = (h * stride) + kh - padH;
            int inW = (w * stride) + kw - padW;

            if (inH >= 0 && inH < inputH && inW >= 0 && inW < inputW) {

                int baseInputIndex = (inH * inputW + inW) * Cin;
                int deltaIndex = (h * W + w) * Cout + f;
                float deltaVal = d[deltaIndex];

                if (c_start < Cin) {
                    sum0 += input_Data[baseInputIndex + c_start] * deltaVal;
                }
                if (c_start + 1 < Cin) {
                    sum1 += input_Data[baseInputIndex + c_start + 1] * deltaVal;
                }
                if (c_start + 2 < Cin) {
                    sum2 += input_Data[baseInputIndex + c_start + 2] * deltaVal;
                }
                if (c_start + 3 < Cin) {
                    sum3 += input_Data[baseInputIndex + c_start + 3] * deltaVal;
                }
            }
        }
    }

    // Write results back
    if (c_start < Cin) {
        wg[kernelRowOffset * Cin + c_start] += sum0;
    }
    if (c_start + 1 < Cin) {
        wg[kernelRowOffset * Cin + c_start + 1] += sum1;
    }
    if (c_start + 2 < Cin) {
        wg[kernelRowOffset * Cin + c_start + 2] += sum2;
    }
    if (c_start + 3 < Cin) {
        wg[kernelRowOffset * Cin + c_start + 3] += sum3;
    }
}
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

__kernel void accumulateTransConvKernelGrads(
    __global const float* activation_outputs,
    __global const float* deltas,
    __global float* weightGrads,
    const int iH,
    const int iW,
    const int iD,
    const int oH,
    const int oW,
    const int filters,
    const int kh_size,
    const int kw_size,
    const int strides,
    const int padTop,
    const int padLeft
) {
    /*
     * One work item computes one (filter, ky, kx, input-channel)
     * weight gradient.
     *
     * Weight layout:
     *   [filter][ky][kx][input_channel]
     *
     * Activation layout:
     *   [y][x][input_channel]
     *
     * Delta layout:
     *   [y][x][filter]
     */

    int filter = get_global_id(0);
    int ky = get_global_id(1);

    int z = get_global_id(2);

    /*
     * The host launches kw_size * iD work items on dimension 2.
     * Decode that into kernel-x and input-channel.
     */
    int kx = z / iD;
    int channel = z % iD;

    if (filter >= filters ||
        ky >= kh_size ||
        kx >= kw_size ||
        channel >= iD) {
        return;
    }

    float sum = 0.0f;
    for (int iy = 0; iy < iH; iy++) {
        int oy = iy * strides + ky - padTop;

        if (oy < 0 || oy >= oH) {
            continue;
        }

        for (int ix = 0; ix < iW; ix++) {
            int ox = ix * strides + kx - padLeft;

            if (ox < 0 || ox >= oW) {
                continue;
            }

            int activationIndex = (iy * iW + ix) * iD + channel;

            int deltaIndex = (oy * oW + ox) * filters + filter;

            sum += activation_outputs[activationIndex] * deltas[deltaIndex];
        }
    }

    int gradIndex = ((filter * kh_size + ky) * kw_size + kx) * iD + channel;

    weightGrads[gradIndex] += sum;
}

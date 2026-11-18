__kernel void accumulateConvWeightandBiasGrads(
    __global const float* input_Data,
    __global const float* d,
    __global float* wg,
    __global float* bg,
    const int inputH,
    const int inputW,
    const int Cin,
    const int H,
    const int W,
    const int Cout,
    const int numFilters,
    const int Kh,
    const int Kw,
    const int padH,
    const int padW,
    const int stride
) {

    int f = get_global_id(0);
    int kh = get_global_id(1);
    int z = get_global_id(2);
    int kw = z / Cin;
    int c = z % Cin;

    if (f >= numFilters || f >= Cout ||
        kh >= Kh ||
        kw >= Kw ||
        c >= Cin) {
        return;
    }

    float sum = 0.0f;
    for (int h = 0; h < H; h++) {
        int inH = (h * stride) + kh - padH;
        if (inH < 0 || inH >= inputH) {
            continue;
        }

        for (int w = 0; w < W; w++) {
            int inW = (w * stride) + kw - padW;
            if (inW < 0 || inW >= inputW) {
                continue;
            }

            int inputIndex = (inH * inputW + inW) * Cin + c;
            int deltaIndex = (h * W + w) * Cout + f;
            sum += input_Data[inputIndex] * d[deltaIndex];
        }
    }

    int gradIndex = ((f * Kh + kh) * Kw + kw) * Cin + c;
    wg[gradIndex] += sum;

    if (kh == 0 && z == 0) {
        float biasSum = 0.0f;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int deltaIndex = (h * W + w) * Cout + f;
                biasSum += d[deltaIndex];
            }
        }
        bg[f] += biasSum;
    }
}

__kernel void accumulateTransConvWeightAndBiasGrads(
    __global const float* activation_outputs,
    __global const float* deltas,
    __global float* weightGrads,
    __global float* biasGrads,
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
    int filter = get_global_id(0);
    int ky = get_global_id(1);
    int z = get_global_id(2);

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

    if (ky == 0 && z == 0) {
        float biasSum = 0.0f;
        for (int oy = 0; oy < oH; oy++) {
            for (int ox = 0; ox < oW; ox++) {
                int deltaIndex = (oy * oW + ox) * filters + filter;
                biasSum += deltas[deltaIndex];
            }
        }
        biasGrads[filter] += biasSum;
    }
}

__kernel void transConv(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    const int inputH,
    const int inputW,
    const int inputD,
    const int outputH,
    const int outputW,
    const int filters,
    const int kernelH,
    const int kernelW,
    const int depth,
    const int strides,
    const int padTop,
    const int padLeft
) {
    int oy = get_global_id(0);
    int ox = get_global_id(1);
    int filter = get_global_id(2);

    if (oy >= outputH || ox >= outputW || filter >= filters) return;

    float sum = biases[filter];
    int outIndex = (oy * outputW + ox) * filters + filter;

    for (int iy = 0; iy < inputH; iy++) {
        for (int ix = 0; ix < inputW; ix++) {
            int inputBase = (iy * inputW + ix) * inputD;

            for (int ky = 0; ky < kernelH; ky++) {
                int mappedY = iy * strides + ky - padTop;
                if (mappedY != oy) continue;

                for (int kx = 0; kx < kernelW; kx++) {
                    int mappedX = ix * strides + kx - padLeft;
                    if (mappedX != ox) continue;

                    int weightBase = ((filter * kernelH + ky) * kernelW + kx) * depth;

                    for (int c = 0; c < depth; c++) {
                        sum += input[inputBase + c] * weights[weightBase + c];
                    }
                }
            }
        }
    }

    output[outIndex] = sum;
}

__kernel void transConvBackward(
    __global const float* delta,
    __global const float* weights,
    __global float* deltaInput,
    const int inputH,
    const int inputW,
    const int inputD,
    const int outputH,
    const int outputW,
    const int outputD,
    const int filters,
    const int kernelH,
    const int kernelW,
    const int depth,
    const int strides,
    const int padTop,
    const int padLeft
) {
    int iy = get_global_id(0);
    int ix = get_global_id(1);
    int c = get_global_id(2);

    if (iy >= inputH || ix >= inputW || c >= inputD) return;

    int inputBase = (iy * inputW + ix) * inputD + c;
    float sum = 0.0f;

    for (int ky = 0; ky < kernelH; ky++) {
        int oy = iy * strides + ky - padTop;

        if (oy < 0 || oy >= outputH) continue;

        for (int kx = 0; kx < kernelW; kx++) {
            int ox = ix * strides + kx - padLeft;

            if (ox < 0 || ox >= outputW) continue;

            for (int filter = 0; filter < filters; filter++) {
                int deltaIndex = (oy * outputW + ox) * outputD + filter;
                int weightBase = ((filter * kernelH + ky) * kernelW + kx) * depth;

                sum += delta[deltaIndex] * weights[weightBase + c];
            }
        }
    }

    deltaInput[inputBase] = sum;
}

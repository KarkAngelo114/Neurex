__kernel void apply_padding(
    __global const float* input,
    __global float* output,
    int inputH,
    int inputW,
    int channels,
    int padTop,
    int padBottom,
    int padLeft,
    int padRight,
    int newH,
    int newW
) {
    // Get global work item IDs for 3D execution
    int h = get_global_id(0);
    int w = get_global_id(1);
    int c = get_global_id(2);

    // Boundary check
    if (h >= newH || w >= newW || c >= channels) {
        return;
    }

    // Calculate output index
    int outIdx = (h * newW + w) * channels + c;

    // Check if this position is within the padded region
    if (h < padTop || h >= (padTop + inputH) || w < padLeft || w >= (padLeft + inputW)) {
        // Padding area - set to 0
        output[outIdx] = 0.0f;
    } else {
        // Data area - copy from input
        int srcH = h - padTop;
        int srcW = w - padLeft;
        int inIdx = (srcH * inputW + srcW) * channels + c;
        output[outIdx] = input[inIdx];
    }
}

__kernel void dilate(
    __global const float* input,
    __global float* dilated,
    const int H,
    const int W,
    const int C,
    const int stride,
    const int dilatedH,
    const int dilatedW
) {

    int c = get_global_id(0);
    int h = get_global_id(1);
    int w = get_global_id(2);

    if (c >= C || h >= H || w >= W) return;

    int srcIdx = (h * W + w) * C + c;
    int dilatedHIdx = h * stride;
    int dilatedWIdx = w * stride;
    int dstIdx = (dilatedHIdx * dilatedW + dilatedWIdx) * C + c;
    dilated[dstIdx] = (input[srcIdx] != 0.0f) ? input[srcIdx] : 0.0f; 

}
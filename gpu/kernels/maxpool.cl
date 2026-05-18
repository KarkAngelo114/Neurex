__kernel void maxpool(
    __global const float* inputTensor,
    __global float* outputTensor,
    __global int* maxIndexTensor,
    const int inputH,
    const int inputW,
    const int inputD,
    const int poolH,
    const int poolW,
    const int outputH,
    const int outputW,
    const int outputD,
    const int strides
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int d = get_global_id(2);

    if (i >= outputH || j >= outputW || d >= outputD) return;

    float maxVal = -INFINITY;
    int maxIdx = -1;

    int startH = i * strides;
    int startW = j * strides;

    for (int ph = 0; ph < poolH; ph++) {
        for (int pw = 0; pw < poolW; pw++) {

            int currH = startH + ph;
            int currW = startW + pw;

            if (currH < inputH && currW < inputW) {
                int idx = (currH * inputW * inputD) + (currW * inputD) + d;
                float val = inputTensor[idx];

                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = idx;
                }
            }
        }
    }

    int outIdx = (i * outputW * outputD) + (j * outputD) + d;

    outputTensor[outIdx] = (maxVal == -INFINITY) ? 0.0f : maxVal;
    maxIndexTensor[outIdx] = maxIdx;
}


__kernel void maxpooldelta(
    __global const float* delta,
    __global const int* indices,
    __global float* outputTensor,
    const int size
) {
    int i = get_global_id(0);

    if (i >= size) return;

    int idx = indices[i];
    if (idx >= 0) outputTensor[idx] += delta[i];
}
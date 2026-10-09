__kernel void accumulateWeightsAndBiasGradsDense (
    __global const float* activations,
    __global const float* deltas,
    __global float* weightGrads,
    __global float* biasGrads,
    const int inputSize,
    const int outputSize
) {

    int i = get_global_id(0); // input size
    int j = get_global_id(1); // output size

    if (i < inputSize && j < outputSize) {
        float input = activations[i];
        int offset = i * outputSize;
        weightGrads[offset + j] += input * deltas[j];
        biasGrads[j] += deltas[j];
    }

}
__kernel void spe(
    __global const float* input,
    const int embeddingDim,
    const int sequenceLength,
    const int size,
    __global float* output
) {
    int index = get_global_id(0);

    if (index >= size) return;

    int position = index / embeddingDim;
    int dimension = index % embeddingDim;

    if (position >= sequenceLength) return;

    int pairIndex = dimension / 2;
    float exponent = (2.0f * (float)pairIndex) / (float)embeddingDim;
    float angle = (float)position / pow(10000.0f, exponent);

    float positionalValue = (dimension % 2 == 0) ? sin(angle) : cos(angle);

    output[index] = input[index] + positionalValue;
}
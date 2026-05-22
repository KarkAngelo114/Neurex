__kernel void getEmbeddings(
    __global const int* tokenArray,
    __global const float* lookup_table,
    __global float* output,
    const int embeddingDim,
    const int sequence_length
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= sequence_length || j >= embeddingDim) return;

    int tokenID = tokenArray[i];
    int startIdx = tokenID * embeddingDim;

    output[i * embeddingDim + j] = lookup_table[startIdx + j];

}

__kernel void returnEmbeddings(
    __global const int* tokenArray,
    __global const float* deltaData,
    __global float* gradsData,
    const int embeddingDim,
    const int sequence_length
) {

    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= sequence_length || j >= embeddingDim) return;

    int tokenId = tokenArray[i];

    // skip PAD tokens (ID 0)
    if (tokenId == 0) return;

    int gradOffset = tokenId * embeddingDim;
    int deltaOffset = i * embeddingDim;

    gradsData[gradOffset + j] += deltaData[deltaOffset + j];

}
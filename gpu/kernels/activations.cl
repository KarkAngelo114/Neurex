
// sigmoid
__kernel void sigmoid(__global float* input, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    input[i] = 1.0f / (1.0f + exp(-input[i]));
}

// relu
__kernel void relu(__global float* input, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    input[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

// tanh
__kernel void Tanh(__global float* input, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    input[i] = tanh(input[i]);
}

// softmax
__kernel void softmax(__global float* input,const float maxVal,const float sum,const int inputSize) {
    
    int i = get_global_id(0);

    if (i >= inputSize) return;

    input[i] = exp(input[i] - maxVal) / sum;
}

//====================== derivatives ==============================//
__kernel void drelu(__global float* input, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    input[i] = input[i] > 0.0f ? 1.0f : 0.0f;
}

__kernel void dsigmoid(__global float* input, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    float sigmoidOutput = 1.0f / (1.0f + exp(-input[i]));
    input[i] = sigmoidOutput * (1.0f - sigmoidOutput);

}

__kernel void dtanh(__global float* input, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    float tanhOutput = tanh(input[i]);

    input[i] = 1.0f - (tanhOutput * tanhOutput);
}

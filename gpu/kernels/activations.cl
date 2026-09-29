// linear
__kernel void linear(
    __global const float* input,
    __global float* output,
    const int inputSize
) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    // write to output buffer
    output[i] = input[i];
}

// relu
__kernel void relu(
    __global const float* input, 
    __global float* output,
    const int inputSize
) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}


// sigmoid
__kernel void sigmoid(__global const float* input, __global float* output, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    output[i] = 1.0f / (1.0f + exp(-input[i]));
}

// tanh
__kernel void Tanh(__global const float* input, __global float* output, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    output[i] = tanh(input[i]);
}

// softmax
__kernel void softmax(__global const float* input, __global float* output, const float maxVal, const float sum, const int inputSize) {
    
    int i = get_global_id(0);

    if (i >= inputSize) return;

    output[i] = exp(input[i] - maxVal) / sum;
}

//====================== derivatives ==============================//
__kernel void drelu(__global const float* input, __global float* output, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    output[i] = input[i] > 0.0f ? 1.0f : 0.0f;
}

__kernel void dsigmoid(__global const float* input, __global float* output, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    float sigmoidOutput = 1.0f / (1.0f + exp(-input[i]));
    output[i] = sigmoidOutput * (1.0f - sigmoidOutput);

}

__kernel void dtanh(__global const float* input, __global float* output, const int inputSize) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    float tanhOutput = tanh(input[i]);

    output[i] = 1.0f - (tanhOutput * tanhOutput);
}

__kernel void dlinear(
    __global const float* input,
    __global float* output,
    const int inputSize
) {
    int i = get_global_id(0);

    if (i >= inputSize) return;

    // write to output buffer
    output[i] = 1.0f;
}

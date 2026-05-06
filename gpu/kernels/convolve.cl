__kernel void convolve(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output_tensor_template,
    const int strides,
    const int output_height,
    const int output_width,
    const int num_filters,
    const int kernel_height,
    const int kernel_width,
    const int depth,
    const int input_height,
    const int input_width
) {
    int oh = get_global_id(0);
    int ow = get_global_id(1);
    int f  = get_global_id(2);

    if (ow >= output_width || oh >= output_height || f >= num_filters)
        return;

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_height; kh++) {
        for (int kw = 0; kw < kernel_width; kw++) {
            for (int c = 0; c < depth; c++) {

                int inY = (oh * strides) + kh;
                int inX = (ow * strides) + kw;

                if (inY < input_height && inX < input_width) {
                    int input_idx  = ((inY * input_width + inX) * depth + c);
                    int kernel_idx = (((f * kernel_height + kh) * kernel_width + kw) * depth + c);

                    sum += input[input_idx] * weights[kernel_idx];
                }
            }
        }
    }

    int outIndex = ((oh * output_width + ow) * num_filters + f);
    output_tensor_template[outIndex] = sum + biases[f];
}
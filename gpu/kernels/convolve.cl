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
    int gid = get_global_id(0);

    int total = output_height * output_width * num_filters;
    if (gid >= total) return;

    // Decode gid → (oh, ow, f)
    int f  = gid % num_filters;
    int ow = (gid / num_filters) % output_width;
    int oh = gid / (num_filters * output_width);

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
__kernel void sgd(
    __global float* params,
    __global const float* grads,
    const float lr,
    const int paramSize
) {

    int i = get_global_id(0);

    if (i >= paramSize) return;

    params[i] -= lr * grads[i];

}

__kernel void adam(
    __global float* params,
    __global const float* gradient,
    __global float* stateM,
    __global float* stateV,
    const int stateT,
    const float learning_rate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int paramSize
) {

    int i = get_global_id(0);

    if (i >= paramSize) return;

    float grad = gradient[i];

    stateM[i] = beta1 * stateM[i] + (1 - beta1) * grad;
    stateV[i] = beta2 * stateV[i] + (1 - beta2) * grad * grad;

    float mHat = stateM[i] / (1.0f - pow(beta1, stateT));
    float vHat = stateV[i] / (1.0f - pow(beta2, stateT));

    params[i] -= learning_rate * mHat / (sqrt(vHat) + epsilon);

}
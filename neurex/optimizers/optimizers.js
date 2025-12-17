
const SGD = (onGPU, params, grads, state, lr) => {

    if (Array.isArray(params[0])) {
        for (let i = 0; i < params.length; i++) {
            for (let j = 0; j < params[i].length; j++) {
                params[i][j] -= lr * grads[i][j];
            }
        }
    } else {
        for (let i = 0; i < params.length; i++) {
            params[i] -= lr * grads[i];
        }
    }
    return state;
};

const Adam = (onGPU, params, grads, state, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {

    if (!state.m) state.m = Array.isArray(params[0])
        ? params.map(row => Array(row.length).fill(0))
        : Array(params.length).fill(0);

    if (!state.v) state.v = Array.isArray(params[0])
        ? params.map(row => Array(row.length).fill(0))
        : Array(params.length).fill(0);

    state.t = (state.t || 0) + 1;

    if (Array.isArray(params[0])) {
        for (let i = 0; i < params.length; i++) {
            for (let j = 0; j < params[i].length; j++) {
                const g = grads[i][j];
                state.m[i][j] = beta1 * state.m[i][j] + (1 - beta1) * g;
                state.v[i][j] = beta2 * state.v[i][j] + (1 - beta2) * g * g;

                const mHat = state.m[i][j] / (1 - Math.pow(beta1, state.t));
                const vHat = state.v[i][j] / (1 - Math.pow(beta2, state.t));

                params[i][j] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    } else {
        for (let i = 0; i < params.length; i++) {
            const g = grads[i];
            state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
            state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;

            const mHat = state.m[i] / (1 - Math.pow(beta1, state.t));
            const vHat = state.v[i] / (1 - Math.pow(beta2, state.t));

            params[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
        }
    }
    return state;
};

const AdaGrad = (onGPU, params, grads, state, lr, epsilon = 1e-8) => {

    if (!state.accum) state.accum = Array(params.length).fill(0);
    for (let i = 0; i < params.length; i++) {
        state.accum[i] += grads[i] * grads[i];
        params[i] -= lr * grads[i] / (Math.sqrt(state.accum[i]) + epsilon);
    }
    return state;
};

const RMSprop = (onGPU, params, grads, state, lr, beta = 0.9, epsilon = 1e-8) => {

    if (!state.accum) state.accum = Array(params.length).fill(0);
    for (let i = 0; i < params.length; i++) {
        state.accum[i] = beta * state.accum[i] + (1 - beta) * grads[i] * grads[i];
        params[i] -= lr * grads[i] / (Math.sqrt(state.accum[i]) + epsilon);
    }
    return state;
};

const Adadelta = (onGPU, params, grads, state, rho = 0.95, epsilon = 1e-6) => {

    if (!state.Eg2) state.Eg2 = Array(params.length).fill(0);
    if (!state.Edx2) state.Edx2 = Array(params.length).fill(0);

    for (let i = 0; i < params.length; i++) {
        state.Eg2[i] = rho * state.Eg2[i] + (1 - rho) * grads[i] * grads[i];
        const dx = - (Math.sqrt(state.Edx2[i] + epsilon) / Math.sqrt(state.Eg2[i] + epsilon)) * grads[i];
        params[i] += dx;
        state.Edx2[i] = rho * state.Edx2[i] + (1 - rho) * dx * dx;
    }

    return state;
};

module.exports = {
    sgd: SGD,
    adam: Adam,
    adagrad: AdaGrad,
    rmsprop: RMSprop,
    adadelta: Adadelta,
};
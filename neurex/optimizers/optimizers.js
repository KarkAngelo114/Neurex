

// const SGD = (params, grads, state, lr) => {
//     // params: array of weights/biases to update
//     // grads: array of gradients (same shape as params)
//     // lr: learning rate
//     for (let i = 0; i < params.length; i++) {
//         params[i] -= lr * grads[i];
//     }
//     return state;
// };

const SGD = (params, grads, state, lr) => {
    // Check if params is a 2D array (for weights) or a 1D array (for biases)
    if (Array.isArray(params[0])) { // It's a 2D array (weights)
        for (let i = 0; i < params.length; i++) { // Iterate over rows
            for (let j = 0; j < params[i].length; j++) { // Iterate over columns/individual weights
                params[i][j] -= lr * grads[i][j]; // Update individual weight
            }
        }
    } else { // It's a 1D array (biases)
        for (let i = 0; i < params.length; i++) {
            params[i] -= lr * grads[i]; // Update individual bias
        }
    }
    return state;
};

// const Adam = (params, grads, state, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {
//     // state: {m: [], v: [], t: int}
//     if (!state.m) state.m = Array(params.length).fill(0);
//     if (!state.v) state.v = Array(params.length).fill(0);
//     state.t = (state.t || 0) + 1;

//     for (let i = 0; i < params.length; i++) {
//         state.m[i] = beta1 * state.m[i] + (1 - beta1) * grads[i];
//         state.v[i] = beta2 * state.v[i] + (1 - beta2) * grads[i] * grads[i];
//         const mHat = state.m[i] / (1 - Math.pow(beta1, state.t));
//         const vHat = state.v[i] / (1 - Math.pow(beta2, state.t));
//         params[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
//     }
//     return state;
// };

const Adam = (params, grads, state, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {
    if (Array.isArray(params[0])) { // It's a 2D array (weights)
        // Initialize internal state as 2D arrays based on params structure
        if (!state.m) state.m = params.map(row => Array(row.length).fill(0));
        if (!state.v) state.v = params.map(row => Array(row.length).fill(0));
        state.t = (state.t || 0) + 1;

        for (let i = 0; i < params.length; i++) { // Iterate over rows
            for (let j = 0; j < params[i].length; j++) { // Iterate over columns/individual weights
                state.m[i][j] = beta1 * state.m[i][j] + (1 - beta1) * grads[i][j];
                state.v[i][j] = beta2 * state.v[i][j] + (1 - beta2) * grads[i][j] * grads[i][j];
                const mHat = state.m[i][j] / (1 - Math.pow(beta1, state.t));
                const vHat = state.v[i][j] / (1 - Math.pow(beta2, state.t));
                params[i][j] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    } else { // It's a 1D array (biases)
        // Initialize internal state as 1D arrays
        if (!state.m) state.m = Array(params.length).fill(0);
        if (!state.v) state.v = Array(params.length).fill(0);
        state.t = (state.t || 0) + 1;

        for (let i = 0; i < params.length; i++) {
            state.m[i] = beta1 * state.m[i] + (1 - beta1) * grads[i];
            state.v[i] = beta2 * state.v[i] + (1 - beta2) * grads[i] * grads[i];
            const mHat = state.m[i] / (1 - Math.pow(beta1, state.t));
            const vHat = state.v[i] / (1 - Math.pow(beta2, state.t));
            params[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
        }
    }
    return state;
};

const AdaGrad = (params, grads, state, lr, epsilon = 1e-8) => {
    // state: {accum: []}
    if (!state.accum) state.accum = Array(params.length).fill(0);
    for (let i = 0; i < params.length; i++) {
        state.accum[i] += grads[i] * grads[i];
        params[i] -= lr * grads[i] / (Math.sqrt(state.accum[i]) + epsilon);
    }
    return state;
};

const RMSprop = (params, grads, state, lr, beta = 0.9, epsilon = 1e-8) => {
    // state: {accum: []}
    if (!state.accum) state.accum = Array(params.length).fill(0);
    for (let i = 0; i < params.length; i++) {
        state.accum[i] = beta * state.accum[i] + (1 - beta) * grads[i] * grads[i];
        params[i] -= lr * grads[i] / (Math.sqrt(state.accum[i]) + epsilon);
    }
    return state;
}

const Adadelta = (params, grads, state, rho = 0.95, epsilon = 1e-6) => {
    // state: {Eg2: [], Edx2: []}
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
}
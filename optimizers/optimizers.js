
const { flattenAll, getShape, reshape } = require('../utils');

const SGD = (params, grads, state = {}, lr) => {

    const flatten_params = flattenAll(params);
    const flatten_grads = flattenAll(grads);
    const params_shape = getShape(params);

    if (flatten_params.length !== flatten_grads.length) {
        throw new Error("SGD: Params and grads size mismatch");
    }

    for (let i = 0; i < flatten_params.length; i++) {
        flatten_params[i] -= lr * flatten_grads[i];
    }

    return {
        params: reshape(flatten_params, params_shape),
        state: state
    };
};


// const Adam = (params, grads, state, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {
//     if (!state.m) state.m = Array.isArray(params[0])
//         ? params.map(row => Array(row.length).fill(0))
//         : Array(params.length).fill(0);

//     if (!state.v) state.v = Array.isArray(params[0])
//         ? params.map(row => Array(row.length).fill(0))
//         : Array(params.length).fill(0);

//     state.t = (state.t || 0) + 1;

//     if (Array.isArray(params[0])) {
//         for (let i = 0; i < params.length; i++) {
//             for (let j = 0; j < params[i].length; j++) {
//                 const g = grads[i][j];
//                 state.m[i][j] = beta1 * state.m[i][j] + (1 - beta1) * g;
//                 state.v[i][j] = beta2 * state.v[i][j] + (1 - beta2) * g * g;

//                 const mHat = state.m[i][j] / (1 - Math.pow(beta1, state.t));
//                 const vHat = state.v[i][j] / (1 - Math.pow(beta2, state.t));

//                 params[i][j] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
//             }
//         }
//     } else {
//         for (let i = 0; i < params.length; i++) {
//             const g = grads[i];
//             state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
//             state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;

//             const mHat = state.m[i] / (1 - Math.pow(beta1, state.t));
//             const vHat = state.v[i] / (1 - Math.pow(beta2, state.t));

//             params[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
//         }
//     }


//     return state;
// };


const Adam = (params, grads, state = {}, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {

    const flatParams = flattenAll(params);
    const flatGrads  = flattenAll(grads);
    const shape      = getShape(params);

    if (flatParams.length !== flatGrads.length) {
        throw new Error("Adam: Params and grads size mismatch");
    }

    // Lazy state initialization
    if (!state.m) {
        state.m = new Array(flatParams.length).fill(0);
        state.v = new Array(flatParams.length).fill(0);
        state.t = 0;
    }

    state.t += 1;

    for (let i = 0; i < flatParams.length; i++) {

        const g = flatGrads[i];

        state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;

        const mHat = state.m[i] / (1 - Math.pow(beta1, state.t));
        const vHat = state.v[i] / (1 - Math.pow(beta2, state.t));

        flatParams[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
    }

    return {
        params: reshape(flatParams, shape),
        state:state
    };
};


module.exports = {
    sgd: SGD,
    adam: Adam,
};

const { ApplySGD, ApplyAdam } = require('../core/bindings');


const SGD = (params, grads, state = {}, lr) => {
    if (params.length !== grads.length) {
        throw new Error("SGD: Params and grads size mismatch");
    }

    return {
        params: ApplySGD(params, grads, lr),
        state: state
    };
};

const Adam = (params, grads, state = {}, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {
    

    if (params.length !== grads.length) {
        throw new Error("Adam: Params and grads size mismatch");
    }

    // Lazy state initialization
    if (!state.m) {
        state.m = new Float32Array(params.length);
        state.v = new Float32Array(params.length);
        state.t = 0;
    }

    state.t += 1;

    const res = ApplyAdam(params, grads, lr, state.m, state.v, state.t, epsilon, beta1, beta2);

    state.m = res.m;
    state.v = res.v;

    return {
        params: res.params,
        state:state
    };
};


module.exports = {
    sgd: SGD,
    adam: Adam,
};

const { ApplySGD, ApplyAdam } = require('../core/bindings');
const { flattenAll, getShape, reshape } = require('../utils');

const SGD = (params, grads, state = {}, lr) => {

    const flatten_params = flattenAll(params);
    const flatten_grads = flattenAll(grads);
    const params_shape = getShape(params);

    if (flatten_params.length !== flatten_grads.length) {
        throw new Error("SGD: Params and grads size mismatch");
    }

    return {
        params: reshape(ApplySGD(flatten_params, flatten_grads, lr), params_shape),
        state: state
    };
};

const Adam = (params, grads, state = {}, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {

    const flatParams = flattenAll(params);
    const flatGrads  = flattenAll(grads);
    const shape      = getShape(params);

    if (flatParams.length !== flatGrads.length) {
        throw new Error("Adam: Params and grads size mismatch");
    }

    // Lazy state initialization
    if (!state.m) {
        state.m = Array.from({length: flatParams.length}).fill(0);
        state.v = Array.from({length: flatParams.length}).fill(0);
        state.t = 0;
    }

    state.t += 1;

    const res = ApplyAdam(flatParams, flatGrads, state.m, state.v, state.t, lr, beta1, beta2, epsilon);

    state.m = res.m.map(m => !Number.isFinite(m) ? 0 : m); // update first momentum, check if there are Infinity values
    state.v = res.v.map(v => !Number.isFinite(v) ? 0 : v); // update second momentum, check if there are Infinity values

    return {
        params: reshape(res.params, shape),
        state:state
    };
};


module.exports = {
    sgd: SGD,
    adam: Adam,
};
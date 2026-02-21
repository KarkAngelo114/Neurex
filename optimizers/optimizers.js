
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

    // if (res.m.some(isNaN) || !Number.isFinite(res.m)) {
    //     console.log();
    //     console.log(res.m)
    //     throw new Error('state.m has Infinity error or NaNs');
    // }

    // if (res.v.some(isNaN) || !Number.isFinite(res.v)) {
    //     console.log();
    //     console.log(res.v)
    //     throw new Error('state.v has Infinity error or NaNs');
    // }

    // state.m = res.m.map(m => !Number.isFinite(m) ? 0 : m); // update first momentum, check if there are Infinity values
    // state.v = res.v.map(v => !Number.isFinite(v) ? 0 : v); // update second momentum, check if there are Infinity values

    res.m.forEach(m => {
        if (!Number.isFinite(m) || isNaN(m)) throw new Error(m) 
    });

    res.v.forEach(v => {
        if (!Number.isFinite(v) || isNaN(v)) throw new Error(v)
    })

    state.m = res.m;
    state.v = res.v;

    return {
        params: reshape(res.params, shape),
        state:state
    };
};


module.exports = {
    sgd: SGD,
    adam: Adam,
};
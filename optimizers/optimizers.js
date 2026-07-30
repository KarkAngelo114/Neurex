
const { ApplySGD, ApplyAdam } = require('../core/bindings');


module.exports = {
    SGD: (momentum = 0.9) => {
        // Name the inner function sgd
        return function SGD(params, grads, state = {}, lr) {
            if (params.length !== grads.length) {
                console.log(grads, params);
                throw new Error("SGD: Params and grads size mismatch");
            }

            if (!state.v) {
                state.v = new Float32Array(params.length);
            }

            const res = ApplySGD(params, grads, state.v, lr, momentum);
            state.v = res.velocity;

            return {
                params: res.params,
                state: state
            };
        };
    },

    Adam: (beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {
        // Name the inner function adam
        return function Adam(params, grads, state = {}, lr) {
            if (params.length !== grads.length) {
                console.log(grads, params);
                throw new Error("Adam: Params and grads size mismatch");
            }

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
                state: state
            };
        };
    }
};
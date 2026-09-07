
const { red, reset } = require('../color-code');
const { ApplySGD, ApplyAdam, ApplyRMSProp } = require('../core/bindings');


module.exports = {
    SGD: (momentum = 0.9) => {
        // Name the inner function sgd
        return function SGD(data) {

            const {params, grads, lr, state: state = {}, previousEpochLoss, current_epoch, batchSize} = data;

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
        return function Adam(data) {

            const {params, grads, lr, state: state = {}, previousEpochLoss, current_epoch, batchSize} = data;

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
    },

    RMSprop: (decayRate = 0.9, epsilon = 1e-8) => {
        
        return function RMSprop(data) {
            const {params, grads, lr, state: state ={}} = data;
            
            if (params.length != grads.length) {
                console.error(`${red}[ERROR]${reset} Parammeter and Gradient sizes does not match.`);
                throw new Error("ERR_PARAM_GRAD_SIZE_MISMATCH");
            }

            if (!state.sqAvg) {
                state.sqAvg = new Float32Array(params.length);
            }

            const res = ApplyRMSProp(params, grads, state.sqAvg, lr, epsilon, decayRate);
            state.sqAvg = res.sqAvg;
            
            return {
                params: res.params,
                state: state
            }
        }
    }
};
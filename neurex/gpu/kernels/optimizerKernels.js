

const { GPU } = require('gpu.js');
const gpu = new GPU();

const sgdKernel = (size) => gpu.createKernel(function (params, grads, lr) {
    return params[this.thread.x] - lr * grads[this.thread.x];
}).setOutput([size]);

const sgdMatrixKernel = (rows, cols) => gpu.createKernel(function (params, grads, lr) {
    return params[this.thread.y][this.thread.x] - lr * grads[this.thread.y][this.thread.x];
}).setOutput([cols, rows]);

function applySGD(params, grads, lr) {
    if (Array.isArray(params[0])) {
        const kernel = sgdMatrixKernel(params.length, params[0].length);
        const result = kernel(params, grads, lr);
        return result.map(row => Array.from(row));
    } else {
        const kernel = sgdKernel(params.length);
        return Array.from(kernel(params, grads, lr));
    }
}

const adamKernel = (size) => gpu.createKernel(function (params, grads, m, v, t, lr, beta1, beta2, epsilon) {
    const g = grads[this.thread.x];
    const mt = beta1 * m[this.thread.x] + (1 - beta1) * g;
    const vt = beta2 * v[this.thread.x] + (1 - beta2) * g * g;

    const mHat = mt / (1 - Math.pow(beta1, t));
    const vHat = vt / (1 - Math.pow(beta2, t));

    return params[this.thread.x] - lr * mHat / (Math.sqrt(vHat) + epsilon);
}).setOutput([size]);

const adamMatrixKernel = (rows, cols) => gpu.createKernel(function (params, grads, m, v, t, lr, beta1, beta2, epsilon) {
    const g = grads[this.thread.y][this.thread.x];
    const mt = beta1 * m[this.thread.y][this.thread.x] + (1 - beta1) * g;
    const vt = beta2 * v[this.thread.y][this.thread.x] + (1 - beta2) * g * g;

    const mHat = mt / (1 - Math.pow(beta1, t));
    const vHat = vt / (1 - Math.pow(beta2, t));

    return params[this.thread.y][this.thread.x] - lr * mHat / (Math.sqrt(vHat) + epsilon);
}).setOutput([cols, rows]);

const applyAdam = (params, grads, state, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {
    if (gpu.mode.toLowerCase() !== 'gpu') {
        throw new Error("[INFO]------- GPU not available. Training on CPU");
    }
    if (!state.m) state.m = Array.isArray(params[0])
        ? params.map(row => Array(row.length).fill(0))
        : Array(params.length).fill(0);

    if (!state.v) state.v = Array.isArray(params[0])
        ? params.map(row => Array(row.length).fill(0))
        : Array(params.length).fill(0);

    state.t = (state.t || 0) + 1;

    if (Array.isArray(params[0])) {
        const kernel = adamMatrixKernel(params.length, params[0].length);
        const result = kernel(params, grads, state.m, state.v, state.t, lr, beta1, beta2, epsilon);
        return result.map(row => Array.from(row));
    } else {
        const kernel = adamKernel(params.length);
        return Array.from(kernel(params, grads, state.m, state.v, state.t, lr, beta1, beta2, epsilon));
    }
}

const adaGradAccumKernel = size => gpu.createKernel(function(accum, grads) {
    return accum[this.thread.x] + grads[this.thread.x] * grads[this.thread.x];
}).setOutput([size]);

const adaGradUpdateKernel = size => gpu.createKernel(function(params, grads, accum, lr, eps) {
    return params[this.thread.x] - lr * grads[this.thread.x] / (Math.sqrt(accum[this.thread.x]) + eps);
}).setOutput([size]);

const applyAdaGrad = (params, grads, state, lr, eps = 1e-8) => {
    if (!state.accum) state.accum = Array(params.length).fill(0);
        // 1) update accum on GPU
        state.accum = Array.from(adaGradAccumKernel(params.length)(state.accum, grads));
        // 2) update params on GPU
        const kernel = adaGradUpdateKernel(params.length);
    return Array.from(kernel(params, grads, state.accum, lr, eps));
}

const rmspropAccumKernel = size => gpu.createKernel(function(accum, grads, beta) {
    return beta * accum[this.thread.x] + (1 - beta) * grads[this.thread.x] * grads[this.thread.x];
}).setOutput([size]);

const rmspropUpdateKernel = size => gpu.createKernel(function(params, grads, accum, lr, eps) {
    return params[this.thread.x] - lr * grads[this.thread.x] / (Math.sqrt(accum[this.thread.x]) + eps);
}).setOutput([size]);

const applyRMSProp = (params, grads, state, lr, beta = 0.9, eps = 1e-8) => {
    if (!state.accum) state.accum = Array(params.length).fill(0);
    // update moving average
    state.accum = Array.from(rmspropAccumKernel(params.length)(state.accum, grads, beta));
    // update params
    const kernel = rmspropUpdateKernel(params.length);
    return Array.from(kernel(params, grads, state.accum, lr, eps));
}

// Eg2 = ρ Eg2 + (1−ρ) grad²
// Δx = − (√Edx2+ε / √Eg2+ε) * grad
// Edx2 = ρ Edx2 + (1−ρ) Δx²
const adadeltaEg2Kernel = size => gpu.createKernel(function(Eg2, grads, rho) {
    return rho * Eg2[this.thread.x] + (1 - rho) * grads[this.thread.x] * grads[this.thread.x];
}).setOutput([size]);

// compute Δx and update params in one go, also return Edx2
const adadeltaUpdateKernel = size => gpu.createKernel(function(params, grads, Eg2, Edx2, rho, eps) {
    const dx = - (Math.sqrt(Edx2[this.thread.x] + eps) / Math.sqrt(Eg2[this.thread.x] + eps)) * grads[this.thread.x];
     // new Edx2:
    this.color(0); // no-op, but we’ll read dx out-of-band
    return params[this.thread.x] + dx;
}).setOutput([size]);

const applyAdadelta = (params, grads, state, rho = 0.95, eps = 1e-6) => {
    // Initialize state
    if (!state.Eg2)  state.Eg2  = Array(params.length).fill(0);
    if (!state.Edx2) state.Edx2 = Array(params.length).fill(0);

    // 1) update Eg2
    state.Eg2 = Array.from(adadeltaEg2Kernel(params.length)(state.Eg2, grads, rho));

    // 2) compute Δx and update params
    const updateKernel = adadeltaUpdateKernel(params.length);
    const newParams = Array.from(updateKernel(params, grads, state.Eg2, state.Edx2, rho, eps));

    // 3) update Edx2 = ρ Edx2 + (1−ρ) dx²
    // We don’t get dx out of the kernel directly, so recompute it here on GPU or CPU.
    // For simplicity, do it on CPU:
    const dx = newParams.map((p, i) => p - params[i]);
    state.Edx2 = state.Edx2.map((old, i) => rho * old + (1 - rho) * dx[i] * dx[i]);

    return newParams;
}
module.exports = {
    applySGD,
    applyAdam,
    applyAdaGrad,
    applyRMSProp,
    applyAdadelta,
    gpuInstance: gpu
};

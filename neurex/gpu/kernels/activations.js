/**
 * GPU-aware activation functions with caching and safe CPU fallback.
 */
const {gpu} = require('../gpu-init');

// Cache kernels by size to avoid re-creating them
const kernelCache = {
  relu: new Map(),
  sigmoid: new Map(),
  tanh: new Map(),
  linear: new Map()
};

// Utility to get or create a kernel safely
function getKernel(type, size, createFn) {
  if (typeof size !== 'number' || size <= 0) {
    throw new Error(`[ACTIVATION ERROR] Invalid size for ${type} kernel: ${size}`);
  }
  if (!kernelCache[type].has(size)) {
    kernelCache[type].set(size, createFn(size));
  }
  return kernelCache[type].get(size);
}

// === GPU Kernels ===
const createReluKernel = size => gpu.createKernel(function (x) {
  return x[this.thread.x] > 0 ? x[this.thread.x] : 0;
}).setOutput([size]);

const createSigmoidKernel = size => gpu.createKernel(function (x) {
  return 1 / (1 + Math.exp(-x[this.thread.x]));
}).setOutput([size]);

const createTanhKernel = size => gpu.createKernel(function (x) {
  const pos = Math.exp(x[this.thread.x]);
  const neg = Math.exp(-x[this.thread.x]);
  return (pos - neg) / (pos + neg);
}).setOutput([size]);

const createLinearKernel = size => gpu.createKernel(function (x) {
  return x[this.thread.x];
}).setOutput([size]);

// === Activation wrappers ===
const relu = (xArray, onGPU) => {
  if (onGPU && xArray?.length > 0) {
    const kernel = getKernel('relu', xArray.length, createReluKernel);
    return kernel(xArray);
  }
  return Math.max(0, xArray);
};

const sigmoid = (xArray, onGPU) => {
  if (onGPU && xArray?.length > 0) {
    const kernel = getKernel('sigmoid', xArray.length, createSigmoidKernel);
    return kernel(xArray);
  }
  return 1 / (1 + Math.exp(-xArray));
};

const tanh = (xArray, onGPU) => {
  if (onGPU && xArray?.length > 0) {
    const kernel = getKernel('tanh', xArray.length, createTanhKernel);
    return kernel(xArray);
  }
  return Math.tanh(xArray);
};

const linear = (xArray, onGPU) => {
  if (onGPU && xArray?.length > 0) {
    const kernel = getKernel('linear', xArray.length, createLinearKernel);
    return kernel(xArray);
  }
  return xArray; // pass-through
};

// Softmax stays CPU for stability
const softmax = (logits, onGPU) => {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
};

// === Derivatives ===
const dreLu = x => x > 0 ? 1 : 0;
const dsigmoid = x => {
  const s = 1 / (1 + Math.exp(-x));
  return s * (1 - s);
};
const dtanh = x => 1 - Math.pow(Math.tanh(x), 2);
const dlinear = () => 1;
const dsoftmax = () => 1;

module.exports = {
  relu,
  sigmoid,
  tanh,
  linear,
  softmax,
  derivatives: {
    relu: dreLu,
    sigmoid: dsigmoid,
    tanh: dtanh,
    linear: dlinear,
    softmax: dsoftmax
  },
};

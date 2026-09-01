const path = require('path');
const { BooleanAvailability } = require('./modeSelector');
let addon = require(path.join(__dirname, "..", "core", "bindings", "prebuilds", `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
let globalWeights = []; // global array of weights
let globalBiases = []; // global array of biases

/**
 * 
 * @param {Array<Float32Array>} weights array of float32array weights
 * @param {Array<Float32Array} biases array of float32array biases
 */
exports.setGlobalParams = (weights, biases) => {
    globalWeights = weights;
    globalBiases = biases;

    if (BooleanAvailability().hasGPU) {
        addon.UploadParams(weights, biases);
    }
}

/** 
 * Use to get paramters from the global store. 
 * @returns {{globalWeights:Float32Array[], globalBiases: Float32Array[], globalOutputTensorTemplate: Float32Array[]}}
*/
exports.getGlobalParams = () => {
    return {
        globalWeights: globalWeights,
        globalBiases: globalBiases,
    }
}
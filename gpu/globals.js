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
    try {
        if (!weights || !biases) {
            throw new Error('[ERROR] No parameters to store....')
        }

        globalWeights = weights;
        globalBiases = biases;

        if (BooleanAvailability().hasGPU) {
            addon.UploadParams(weights, biases);
        }
    }
    catch (e) {
        console.log(`${red}Parameter error${reset}`);
        console.error(e);
        process.exit(1);
    }
}

/** 
 * Use to get paramters from the global store. 
 * @returns {{globalWeights:Float32Array[], globalBiases: Float32Array[], globalOutputTensorTemplate: Float32Array[]}}
*/
exports.getGlobalParams = () => {
    try {

        if (globalWeights.length == 0 || globalBiases.length == 0) {
            throw new Error('[ERROR] No parameters in stored. Call "setParams() first before your custom training loop."')
        }

        return {
            globalWeights: globalWeights,
            globalBiases: globalBiases,
        }
    }
    catch (e) {
        console.log(`\n[GLOBAL STORE ERROR]`);
        console.error(e);
        process.exit(1);
    }
    
}
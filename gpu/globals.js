const path = require('path');
const { BooleanAvailability } = require('./modeSelector');
const { red, reset } = require('../color-code');
let addon = require(path.join(__dirname, "..", "core", "bindings", "prebuilds", `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
const paramStore = new Map();


/**
 * 
 * @param {String} modelID use to refer to model's corresponding param structure
 * @param {Array<Float32Array>} weights array of float32array weights
 * @param {Array<Float32Array} biases array of float32array biases
 */
exports.setGlobalParams = (modelID, weights, biases) => {
    try {

        if (!modelID) {
            throw new Error("[ERROR] No model ID");
        }

        if (!weights || !biases) {
            throw new Error('[ERROR] No parameters to store....')
        }

        paramStore.set(modelID, {weights, biases});

        if (BooleanAvailability().hasGPU) {
            addon.UploadParams(modelID, weights, biases);
        }
    }
    catch (e) {
        console.log(`${red}Parameter error${reset}`);
        console.error(e);
        process.exit(1);
    }
}

/** 
 * @function getGLobalParams Use to get paramters from the global store.
 * @param {String} modelID a generated/saved model ID to be use to refer to the global store 
 * @returns {{globalWeights:Float32Array[], globalBiases: Float32Array[], globalOutputTensorTemplate: Float32Array[]}}
*/
exports.getGlobalParams = (modelID) => {
    try {

        if (!modelID) {
            throw new Error("[ERROR] No model ID");
        }
        
        const entry = paramStore.get(modelID);

        if (!entry || entry.weights.length == 0 || entry.biases.length == 0) {
            throw new Error(`[ERROR] No parameters stored for model "${modelID}". Call "setParams()" first before your custom training loop.`);
        }

        return {
            globalWeights: entry.weights,
            globalBiases: entry.biases,
        }
    }
    catch (e) {
        console.log(`\n[GLOBAL STORE ERROR]`);
        console.error(e);
        process.exit(1);
    }
    
}
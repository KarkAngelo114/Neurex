const path = require('path');
const errFlag = require('../err_flags');
const { globalState } = require('./modeSelector');
const { red, reset } = require('../color-code');
let addon = require(path.join(__dirname, "..", "core", "bindings", "prebuilds", `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
let isUploaded = false;

/**
 * This function only applies when using a GPU. No more global parameter store here on JS side since parameters are owned by parametric layers found in their metadata.
 * @param {String} modelID use to refer to model's corresponding param structure
 * @param {Array<{layerID: string, weights: Float32Array, biases: Float32Array}>} layerParameters
 */
exports.setGlobalParams = (modelID, layerParameters) => {
    try {

        if (!modelID) {
            throw new Error("[ERROR] No model ID");
        }

        if (!Array.isArray(layerParameters) || layerParameters.length === 0) {
            console.error('[ERROR] No parameters to store....');
            throw new Error(errFlag.MODEL_PARAMETER_ERR);
        }

        for (const parameters of layerParameters) {
            if (!parameters.layerID || !(parameters.weights instanceof Float32Array) || !(parameters.biases instanceof Float32Array)) {
                throw new Error("[ERROR] Invalid layer parameter metadata");
            }
        }

        if (globalState().computeBackend == "opencl" && !isUploaded) {
            addon.UploadParams(modelID, layerParameters, globalState().computeBackend);
            isUploaded = true;
        }

    }
    catch (e) {
        console.error(`${red}[GLOBAL STORE ERROR]${reset} Failed to store model parameters:`, e);
        throw e;
    }
}
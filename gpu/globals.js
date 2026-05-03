const path = require('path');
const { BooleanAvailability } = require('./modeSelector');
let addon = require(path.join(__dirname, "..", "core", "bindings", "prebuilds", `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
let globalWeights = []; // global array of weights
let globalBiases = []; // global array of biases
let globalOutputTensorTemplate = []; // global array of output templates used in feedforward only so that no need to create new Flaot32Array each time a layer function is called and to return an output during feedforward. Applies only to layers


exports.setGlobalParams = (weights, biases, outputTemplates) => {
    globalWeights = weights;
    globalBiases = biases;
    globalOutputTensorTemplate = outputTemplates;

    // upload the parameters to C++ side to be use by
    if (addon) {
        addon.setGlobalParams(weights, biases, outputTemplates);
    }

}

exports.getGlobalParams = () => {
    const {hasGPU} = BooleanAvailability;
    if (hasGPU) {
        // call the native functions
        return;
    }
    return {
        globalWeights: globalWeights,
        globalBiases: globalBiases,
        globalOutputTensorTemplate: globalOutputTensorTemplate
    }
}

exports.replaceWeightParamByIndex = (param, pointer) => {
    globalWeights[pointer] = param;
}

exports.replaceBiasParamByIndex = (param, pointer) => {
    globalBiases[pointer] = param;
}
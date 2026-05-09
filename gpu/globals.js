const path = require('path');
const { BooleanAvailability } = require('./modeSelector');
let addon = require(path.join(__dirname, "..", "core", "bindings", "prebuilds", `${process.platform}-${process.arch}`, 'neurex-core-native.node'));
let globalWeights = []; // global array of weights
let globalBiases = []; // global array of biases
let globalOutputTensorTemplate = []; // global array of output templates used in feedforward only so that no need to create new Flaot32Array each time a layer function is called and to return an output during feedforward. Applies only to layers
let globalWeightGrads = []; // global array of weight grads
let globalBiasGrads = []; // global array of bias grads


exports.setGlobalParams = (weights, biases, outputTemplates, weightGrads, biasGrads) => {
    globalWeights = weights;
    globalBiases = biases;
    globalOutputTensorTemplate = outputTemplates;
    globalWeightGrads = weightGrads;
    globalBiasGrads = biasGrads;

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
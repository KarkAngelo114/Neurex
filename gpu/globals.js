const path = require('path');
const { BooleanAvailability } = require('./modeSelector');
let globalWeights = []; // global array of weights
let globalBiases = []; // global array of biases


exports.setGlobalParams = (weights, biases, outputTemplates) => {
    globalWeights = weights;
    globalBiases = biases;
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
let globalWeights = []; // global array of weights
let globalBiases = []; // global array of biases
let globalOutputTensorTemplate = []; // global array of output templates used in feedforward only so that no need to create new Flaot32Array each time a layer function is called and to return an output during feedforward. Applies only to layers


exports.setGlobalParams = (weights, biases, outputTemplates) => {
    globalWeights = weights;
    globalBiases = biases;
    globalOutputTensorTemplate = outputTemplates;

}

exports.getGlobalParams = () => {
    return {
        globalWeights: globalWeights,
        globalBiases: globalBiases,
        globalOutputTensorTemplate: globalOutputTensorTemplate
    }
}

exports.indexWeights = (param, pointer) => {
    globalWeights[pointer] = param;
}
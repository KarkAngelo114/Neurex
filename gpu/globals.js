let globalWeights = [];
let globalBiases = [];


exports.setGlobalParams = (weights, biases) => {
    globalWeights = weights;
    globalBiases = biases;
}

exports.getGlobalParams = () => {
    return {
        globalWeights: globalWeights,
        globalBiases: globalBiases
    }
}

exports.indexWeights = (param, pointer) => {
    globalWeights[pointer] = param;
}
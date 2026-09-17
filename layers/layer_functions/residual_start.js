
const { element_wise_add } = require("../../core/bindings/entry");
const { setResidual, getResidual } = require("./residual")

// residual layers has no params. So, return with defaults and empty arrays. Shapes also must be preserved to pass it to the next layer
const initParams = (size, shape, layer_data) => {

    return {
        updatedSize: size,
        updatedShape: shape,
        weights: [],
        biases: [],
        weightGrads: [],
        biasGrads: [],
        inputShape: shape,
        outputShape: shape,
        paramShape: [],
    }
}


const feedforward = (input, modelID) => {
    // residual start will just pass through the input, but at the same time, it will cache the input to be use by the `residualEnd()`
    setResidual(modelID, input);

    return {
        outputs: input,
        z_values: input,
        incrementor_value:0
    }
}

const projectDeltaBackward = (delta, modelID) => {
    // if the feedforward does caching the input, delta projection does the opposite, it will perform what the `residualEnd` do during feedforward

    const cached = getResidual(modelID); // this must be the delta stored by the `residualEnd` during delta projection

    const output = element_wise_add(delta, cached);

    if (output.some(v => Number.isNaN(v))) throw new Error("RESIDUAL_ADD_ERROR: output contains NaNs.");

    return output;

}


module.exports = {
    initParams,
    feedforward,
    projectDeltaBackward
}
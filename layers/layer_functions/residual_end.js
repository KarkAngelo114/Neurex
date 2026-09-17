
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
    // residualEnd will add the projected output by the previous layers to the stored input in an element-wise manner
    const cached = getResidual(modelID); // this must be the cached input by the `residualStart`

    const output = element_wise_add(cached, input);

    if (output.some(v => Number.isNaN(v))) throw new Error("RESIDUAL_ADD_ERROR: output contains NaNs.");

    return {
        outputs: output,
        z_values: output,
        incrementor_value:0
    }
}

const projectDeltaBackward = (delta, modelID) => {
    // if the feedforward does the adding of cached input and the projected transformed output, delta projection does the opposite, it will perform what the `residualStart` do during feedforward
    setResidual(modelID, delta); // store the projected delta to be use later by the `residualStart` during backpropagation

    return delta; // delta will pass through

}


module.exports = {
    initParams,
    feedforward,
    projectDeltaBackward
}
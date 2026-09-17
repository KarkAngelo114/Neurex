const cache = new Map();

const setResidual = (modelID, tensor) => {
    cache.set(modelID, tensor);
}

const getResidual = (modelID) => {
    const cached = cache.get(modelID);

    if (!cached) {
        console.log("No cached input to return. This is error will occur if you haven't start a residual connection by adding a `residualStart()` in your model.");
        throw new Error("ERR_NO_RESIDUAL_CACHED");
    }

    // removes the cached input, ending the residual connection
    cache.delete(modelID);
    return cached;
}

module.exports = {
    setResidual,
    getResidual
}

const initParams = (size, shape, layer_data) => {
    try {
        const targetShape = layer_data.targetShape;
        const updatedSize = targetShape.reduce((acc, val) => acc * val , 1);
        const incomingSize = shape.reduce((acc, val) => acc * val , 1);

        if (incomingSize != updatedSize) {
            console.error(`[RESHAPE ERROR]------ Incoming shape must match reshape target shape. Expected shape: ${targetShape} or ${updatedSize} | Incoming shape: ${shape} or ${incomingSize}`);
            throw new Error('RESHAPE ERROR');
        }

        return {
            updatedSize: updatedSize,
            updatedShape: targetShape,
            weights: [],
            biases: [],
            weightGrads: [],
            biasGrads: [],
            inputShape: shape,
            outputShape: targetShape,
            paramShape: [],
        }
    }
    catch (error) {
        console.log(error);
        process.exit(1)
    }
}

const feedforward = (input) => {
    return {
        outputs: input,
        z_values: input,
        incrementor_value:0
    }
}



module.exports = {
    initParams,
    feedforward
}
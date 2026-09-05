// this file is for translating layer types to their ONNX equivalent graph or nodes.
// Each translate* function returns plain JS descriptors (no onnx-buf/protobuf
// objects here) so this file stays runtime-agnostic and easy to unit test.
// onnx-exporter.js is responsible for turning these descriptors into real
// onnx-buf NodeProto/TensorProto messages.

// these are helper mappers to map Neurex activation function names to ONNX
const ACTIVATION_TO_ONNX_OP = {
    relu: 'Relu',
    sigmoid: 'Sigmoid',
    tanh: 'Tanh',
    softmax: 'Softmax',
    linear: null, // identity - no activation node is emitted
};

/**
 * Translates a single "Connected Layer" (dense/fully-connected layer) into
 * the ONNX node + initializer descriptors that represent it:
 *   MatMul(input, weight) -> matmul_out
 *   Add(matmul_out, bias) -> add_out           (only if useBias)
 *   Activation(add_out)   -> layer output      (only if activation isn't linear/identity)
 *
 * @param {Object} layer - one entry from Neurex's `this.layers` (must be layer_name === "Connected Layer")
 * @param {Float32Array} weight - this layer's weight tensor, flat, shape [inputSize, outputSize]
 * @param {Float32Array} bias - this layer's bias tensor, shape [outputSize] (may be empty/unused if !useBias)
 * @param {String} inputName - the ONNX tensor name feeding into this layer (previous layer's output, or graph input)
 * @param {Number} layerIndex - this layer's position in the stack, used to build unique, stable tensor/node names
 * @returns {{nodes: Array<Object>, initializers: Array<Object>, outputName: String}}
 *   nodes: ordered list of { name, opType, inputs: string[], outputs: string[] }
 *   initializers: list of { name, dims: number[], data: Float32Array } (weight/bias tensors for this layer)
 *   outputName: the ONNX tensor name this layer produces - feed this in as the next layer's inputName
 */
const translateConnectedLayer = (layer, weight, bias, inputName, layerIndex) => {
    if (layer.layer_name !== 'Connected Layer') {
        throw new Error(`translateConnectedLayer received a layer of type "${layer.layer_name}", expected "Connected Layer"`);
    }

    const [inputSize, outputSize] = layer.weightShape;
    if (!inputSize || !outputSize) {
        throw new Error(`translateConnectedLayer: layer at index ${layerIndex} is missing a valid weightShape ([${layer.weightShape}])`);
    }

    const useBias = layer.useBias ?? true;
    const activationName = layer.activation_function ? layer.activation_function.name : 'linear';
    const onnxActivationOp = ACTIVATION_TO_ONNX_OP[activationName];

    if (onnxActivationOp === undefined) {
        throw new Error(`translateConnectedLayer: activation "${activationName}" (layer index ${layerIndex}) has no ONNX equivalent registered in ACTIVATION_TO_ONNX_OP`);
    }

    const namePrefix = `layer${layerIndex}`;
    const weightName = `${namePrefix}_weight`;
    const biasName = `${namePrefix}_bias`;

    const nodes = [];
    const initializers = [
        { name: weightName, dims: [inputSize, outputSize], data: weight },
    ];

    // MatMul: input [1, inputSize] x weight [inputSize, outputSize] -> [1, outputSize]
    const matmulOutput = `${namePrefix}_matmul_out`;
    nodes.push({
        name: `${namePrefix}_matmul`,
        opType: 'MatMul',
        inputs: [inputName, weightName],
        outputs: [matmulOutput],
    });

    let lastOutput = matmulOutput;

    if (useBias) {
        initializers.push({ name: biasName, dims: [outputSize], data: bias });
        const addOutput = `${namePrefix}_add_out`;
        nodes.push({
            name: `${namePrefix}_add`,
            opType: 'Add',
            inputs: [lastOutput, biasName],
            outputs: [addOutput],
        });
        lastOutput = addOutput;
    }

    if (onnxActivationOp !== null) {
        const activationOutput = `${namePrefix}_activation_out`;
        nodes.push({
            name: `${namePrefix}_${activationName}`,
            opType: onnxActivationOp,
            inputs: [lastOutput],
            outputs: [activationOutput],
        });
        lastOutput = activationOutput;
    }

    return {
        nodes: nodes,
        initializers: initializers,
        outputName: lastOutput,
    };
};

module.exports = {
    translateConnectedLayer,
    ACTIVATION_TO_ONNX_OP,
};
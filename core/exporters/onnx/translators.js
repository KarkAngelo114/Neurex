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

const spatialTranspose = (name, inputName, outputName, permutation) => ({
    name,
    opType: 'Transpose',
    inputs: [inputName],
    outputs: [outputName],
    attributes: [{ name: 'perm', type: 'INTS', value: permutation.map((value) => BigInt(value)) }],
});

const normalizeSpatialValue = (value, fieldName, layerIndex) => {
    const values = Array.isArray(value) ? value : [value, value];
    if (values.length !== 2 || values.some((entry) => !Number.isInteger(entry) || entry <= 0)) {
        throw new Error(`ONNX export: layer at index ${layerIndex} must have a valid ${fieldName} ([height, width])`);
    }
    return values;
};

// Neurex stores kernels as [filter, kernelHeight, kernelWidth, inputChannel].
// ONNX Conv initializers use [outputChannel, inputChannel, kernelHeight, kernelWidth].
const toOnnxConvWeights = (weight, [filters, kernelHeight, kernelWidth, inputChannels], layerIndex) => {
    const expectedLength = filters * kernelHeight * kernelWidth * inputChannels;
    if (!weight || weight.length !== expectedLength) {
        throw new Error(`translateConvLayer: layer at index ${layerIndex} has ${weight ? weight.length : 0} weights; expected ${expectedLength} from weightShape [${filters}, ${kernelHeight}, ${kernelWidth}, ${inputChannels}]`);
    }

    const onnxWeight = new Float32Array(expectedLength);
    for (let filter = 0; filter < filters; filter++) {
        for (let channel = 0; channel < inputChannels; channel++) {
            for (let y = 0; y < kernelHeight; y++) {
                for (let x = 0; x < kernelWidth; x++) {
                    const neurexIndex = ((filter * kernelHeight + y) * kernelWidth + x) * inputChannels + channel;
                    const onnxIndex = ((filter * inputChannels + channel) * kernelHeight + y) * kernelWidth + x;
                    onnxWeight[onnxIndex] = weight[neurexIndex];
                }
            }
        }
    }
    return onnxWeight;
};

// Neurex stores kernels as [filter, kernelHeight, kernelWidth, inputChannel].
// ONNX ConvTranspose initializers use [inputChannel, filter, kernelHeight, kernelWidth].
const toOnnxConvTransposeWeights = (weight, [filters, kernelHeight, kernelWidth, inputChannels], layerIndex) => {
    const expectedLength = filters * kernelHeight * kernelWidth * inputChannels;
    if (!weight || weight.length !== expectedLength) {
        throw new Error(`translateTransConv: layer at index ${layerIndex} has ${weight ? weight.length : 0} weights; expected ${expectedLength} from weightShape [${filters}, ${kernelHeight}, ${kernelWidth}, ${inputChannels}]`);
    }

    const onnxWeight = new Float32Array(expectedLength);
    for (let filter = 0; filter < filters; filter++) {
        for (let channel = 0; channel < inputChannels; channel++) {
            for (let y = 0; y < kernelHeight; y++) {
                for (let x = 0; x < kernelWidth; x++) {
                    const neurexIndex = ((filter * kernelHeight + y) * kernelWidth + x) * inputChannels + channel;
                    const onnxIndex = ((channel * filters + filter) * kernelHeight + y) * kernelWidth + x;
                    onnxWeight[onnxIndex] = weight[neurexIndex];
                }
            }
        }
    }
    return onnxWeight;
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

/**
 *
 * @param {Object} layer - one entry from Neurex's `this.layers` (must be layer_name === "Reshape")
 * @param {Float32Array} weight - unused, present only for signature parity with other translate* fns
 * @param {Float32Array} bias - unused, present only for signature parity with other translate* fns
 * @param {String} inputName - the ONNX tensor name feeding into this layer
 * @param {Number} layerIndex - this layer's position in the stack, used to build unique, stable tensor/node names
 * @returns {{nodes: Array<Object>, initializers: Array<Object>, outputName: String}}
 */
const translateReshape = (layer, weight, bias, inputName, layerIndex) => {
    if (layer.layer_name !== 'Reshape') {
        throw new Error(`translateReshape received a layer of type "${layer.layer_name}", expected "Reshape"`);
    }

    const targetShape = layer.targetShape;
    if (!targetShape || targetShape.length === 0) {
        throw new Error(`translateReshape: layer at index ${layerIndex} is missing a valid targetShape ([${targetShape}])`);
    }

    const namePrefix = `layer${layerIndex}`;
    const shapeName = `${namePrefix}_shape`;
    const outputName = `${namePrefix}_reshape_out`;

    const flatSize = targetShape.reduce((total, dim) => total * dim, 1);
    const shapeData = new BigInt64Array([-1n, BigInt(flatSize)]);

    const initializers = [
        { name: shapeName, dims: [shapeData.length], data: shapeData, dataType: 'INT64' },
    ];

    const nodes = [
        {
            name: `${namePrefix}_reshape`,
            opType: 'Reshape',
            inputs: [inputName, shapeName],
            outputs: [outputName],
        },
    ];

    return {
        nodes,
        initializers,
        outputName,
    };
};

/**
 * Translates a "Layer Normalization" layer into ONNX's built-in
 * LayerNormalization op (opset >= 17). Shape is preserved end to end -
 * layer.weightShape (== [size]) gives the dims for gamma/beta, and the
 * normalization is applied over the last axis (axis = -1), matching
 * Neurex's per-sample, feature-wise normalization in layerNorm.js.
 *
 *   LayerNormalization(input, gamma, beta) -> layer output
 *
 * @param {Object} layer - one entry from Neurex's `this.layers` (must be layer_name === "Layer Normalization"). Uses layer.weightShape (== [size]) for gamma/beta dims - LayerNorm layers don't carry a separate paramShape field.
 * @param {Float32Array} weight - this layer's gamma (scale) tensor, shape [size]
 * @param {Float32Array} bias - this layer's beta (shift) tensor, shape [size]
 * @param {String} inputName - the ONNX tensor name feeding into this layer
 * @param {Number} layerIndex - this layer's position in the stack, used to build unique, stable tensor/node names
 * @returns {{nodes: Array<Object>, initializers: Array<Object>, outputName: String}}
 */
const translateLayerNorm = (layer, weight, bias, inputName, layerIndex) => {
    if (layer.layer_name !== 'Layer Normalization') {
        throw new Error(`translateLayerNorm received a layer of type "${layer.layer_name}", expected "Layer Normalization"`);
    }

    const paramShape = layer.weightShape;

    if (!paramShape || paramShape.length === 0) {
        throw new Error(`translateLayerNorm: layer at index ${layerIndex} is missing a valid weightShape ([${paramShape}])`);
    }

    const expectedSize = paramShape.reduce((a, d) => a * d, 1);
    if (weight.length !== expectedSize || bias.length !== expectedSize) {
        throw new Error(
            `translateLayerNorm: layer at index ${layerIndex} has weightShape [${paramShape}] (expects ${expectedSize} values) ` +
            `but gamma has ${weight.length} and beta has ${bias.length} - refusing to export a mismatched LayerNormalization node`
        );
    }

    const eps = layer.eps ?? 1e-5;
    const namePrefix = `layer${layerIndex}`;
    const gammaName = `${namePrefix}_gamma`;
    const betaName = `${namePrefix}_beta`;
    const outputName = `${namePrefix}_layernorm_out`;

    const initializers = [
        { name: gammaName, dims: paramShape, data: weight },
        { name: betaName, dims: paramShape, data: bias },
    ];

    const nodes = [
        {
            name: `${namePrefix}_layernorm`,
            opType: 'LayerNormalization',
            inputs: [inputName, gammaName, betaName],
            outputs: [outputName],
            attributes: [
                { name: 'axis', type: 'INT', value: -1n },
                { name: 'epsilon', type: 'FLOAT', value: eps },
            ],
        },
    ];

    return {
        nodes,
        initializers,
        outputName,
    };
};

const translateMaxPool = (layer, weight, bias, inputName, layerIndex) => {
    if (layer.layer_name !== 'Max Pooling') {
        throw new Error(`translateMaxPool received a layer of type "${layer.layer_name}", expected "Max Pooling"`);
    }

    const [poolHeight, poolWidth] = normalizeSpatialValue(layer.poolSize, 'poolSize', layerIndex);
    const stride = layer.strides || 1;
    if (!Number.isInteger(stride) || stride <= 0) {
        throw new Error(`translateMaxPool: layer at index ${layerIndex} must have a positive integer stride`);
    }
    const padding = (layer.padding || 'same').toLowerCase();
    if (padding !== 'same' && padding !== 'valid') {
        throw new Error(`translateMaxPool: unsupported padding "${layer.padding}" at layer index ${layerIndex}`);
    }

    const namePrefix = `layer${layerIndex}`;
    const nchwInput = `${namePrefix}_nchw_input`;
    const nchwOutput = `${namePrefix}_nchw_out`;
    const outputName = `${namePrefix}_maxpool_out`;
    const nodes = [
        spatialTranspose(`${namePrefix}_to_nchw`, inputName, nchwInput, [0, 3, 1, 2]),
        {
            name: `${namePrefix}_maxpool`,
            opType: 'MaxPool',
            inputs: [nchwInput],
            outputs: [nchwOutput],
            attributes: [
                { name: 'kernel_shape', type: 'INTS', value: [BigInt(poolHeight), BigInt(poolWidth)] },
                { name: 'strides', type: 'INTS', value: [BigInt(stride), BigInt(stride)] },
                { name: 'auto_pad', type: 'STRING', value: padding === 'same' ? 'SAME_UPPER' : 'VALID' },
            ],
        },
        spatialTranspose(`${namePrefix}_to_nhwc`, nchwOutput, outputName, [0, 2, 3, 1]),
    ];

    return { nodes, initializers: [], outputName };
};

const translateConvLayer = (layer, weight, bias, inputName, layerIndex) => {
    if (layer.layer_name !== 'Convolutional Layer') {
        throw new Error(`translateConvLayer received a layer of type "${layer.layer_name}", expected "Convolutional Layer"`);
    }

    const weightShape = layer.weightShape;
    if (!weightShape || weightShape.length !== 4 || weightShape.some((value) => !Number.isInteger(value) || value <= 0)) {
        throw new Error(`translateConvLayer: layer at index ${layerIndex} is missing a valid weightShape [filters, kernelHeight, kernelWidth, inputChannels]`);
    }
    const [filters, kernelHeight, kernelWidth, inputChannels] = weightShape;
    const [configuredKernelHeight, configuredKernelWidth] = normalizeSpatialValue(layer.kernel_size, 'kernel_size', layerIndex);
    if (configuredKernelHeight !== kernelHeight || configuredKernelWidth !== kernelWidth) {
        throw new Error(`translateConvLayer: kernel_size [${layer.kernel_size}] does not match weightShape [${weightShape}] at layer index ${layerIndex}`);
    }

    const stride = layer.strides || 1;
    if (!Number.isInteger(stride) || stride <= 0) {
        throw new Error(`translateConvLayer: layer at index ${layerIndex} must have a positive integer stride`);
    }
    const padding = (layer.padding || 'same').toLowerCase();
    if (padding !== 'same' && padding !== 'valid') {
        throw new Error(`translateConvLayer: unsupported padding "${layer.padding}" at layer index ${layerIndex}`);
    }

    const useBias = layer.useBias ?? true;
    if (useBias && (!bias || bias.length !== filters)) {
        throw new Error(`translateConvLayer: layer at index ${layerIndex} has ${bias ? bias.length : 0} biases; expected ${filters}`);
    }

    const namePrefix = `layer${layerIndex}`;
    const weightName = `${namePrefix}_conv_weight`;
    const biasName = `${namePrefix}_conv_bias`;
    const nchwInput = `${namePrefix}_nchw_input`;
    const nchwOutput = `${namePrefix}_nchw_out`;
    const nhwcOutput = `${namePrefix}_conv_out`;
    const initializers = [
        { name: weightName, dims: [filters, inputChannels, kernelHeight, kernelWidth], data: toOnnxConvWeights(weight, weightShape, layerIndex) },
    ];
    const convInputs = [nchwInput, weightName];
    if (useBias) {
        initializers.push({ name: biasName, dims: [filters], data: bias });
        convInputs.push(biasName);
    }

    const nodes = [
        spatialTranspose(`${namePrefix}_to_nchw`, inputName, nchwInput, [0, 3, 1, 2]),
        {
            name: `${namePrefix}_conv`,
            opType: 'Conv',
            inputs: convInputs,
            outputs: [nchwOutput],
            attributes: [
                { name: 'kernel_shape', type: 'INTS', value: [BigInt(kernelHeight), BigInt(kernelWidth)] },
                { name: 'strides', type: 'INTS', value: [BigInt(stride), BigInt(stride)] },
                { name: 'auto_pad', type: 'STRING', value: padding === 'same' ? 'SAME_UPPER' : 'VALID' },
            ],
        },
        spatialTranspose(`${namePrefix}_to_nhwc`, nchwOutput, nhwcOutput, [0, 2, 3, 1]),
    ];

    const activationName = layer.activation_function ? layer.activation_function.name : 'linear';
    const activationOp = ACTIVATION_TO_ONNX_OP[activationName];
    if (activationOp === undefined) {
        throw new Error(`translateConvLayer: activation "${activationName}" (layer index ${layerIndex}) has no ONNX equivalent registered in ACTIVATION_TO_ONNX_OP`);
    }
    let outputName = nhwcOutput;
    if (activationOp !== null) {
        outputName = `${namePrefix}_activation_out`;
        nodes.push({
            name: `${namePrefix}_${activationName}`,
            opType: activationOp,
            inputs: [nhwcOutput],
            outputs: [outputName],
        });
    }

    return { nodes, initializers, outputName };
};


const translateTransConv = (layer, weight, bias, inputName, layerIndex) => {
    const weightShape = layer.weightShape;
    const [filters, kernelHeight, kernelWidth, inputChannels] = weightShape;
    const stride = layer.strides || 1;
    const padding = (layer.padding || 'same').toLowerCase();
    const useBias = layer.useBias ?? true;

    if (layer.layer_name !== 'Trans Convolution') {
        throw new Error(`translateTransConv received a layer of type "${layer.layer_name}", expected "Trans Convolution"`);
    }

    if (!weightShape || weightShape.length !== 4 || weightShape.some((value) => !Number.isInteger(value) || value <= 0)) {
        throw new Error(`translateTransConv: layer at index ${layerIndex} is missing a valid weightShape [filters, kernelHeight, kernelWidth, inputChannels]`);
    }
    
    const [configuredKernelHeight, configuredKernelWidth] = normalizeSpatialValue(layer.kernel_size, 'kernel_size', layerIndex);
    if (configuredKernelHeight !== kernelHeight || configuredKernelWidth !== kernelWidth) {
        throw new Error(`translateTransConv: kernel_size [${layer.kernel_size}] does not match weightShape [${weightShape}] at layer index ${layerIndex}`);
    }

    
    if (!Number.isInteger(stride) || stride <= 0) {
        throw new Error(`translateTransConv: layer at index ${layerIndex} must have a positive integer stride`);
    }
    
    if (padding !== 'same' && padding !== 'valid') {
        throw new Error(`translateTransConv: unsupported padding "${layer.padding}" at layer index ${layerIndex}`);
    }

    if (useBias && (!bias || bias.length !== filters)) {
        throw new Error(`translateTransConv: layer at index ${layerIndex} has ${bias ? bias.length : 0} biases; expected ${filters}`);
    }

    const namePrefix = `layer${layerIndex}`;
    const weightName = `${namePrefix}_conv_weight`;
    const biasName = `${namePrefix}_conv_bias`;
    const nchwInput = `${namePrefix}_nchw_input`;
    const nchwOutput = `${namePrefix}_nchw_out`;
    const nhwcOutput = `${namePrefix}_conv_out`;

    const initializers = [
        { name: weightName, dims: [inputChannels, filters, kernelHeight, kernelWidth], data: toOnnxConvTransposeWeights(weight, weightShape, layerIndex) },
    ];

    const convInputs = [nchwInput, weightName];
    if (useBias) {
        initializers.push({ name: biasName, dims: [filters], data: bias });
        convInputs.push(biasName);
    }

    const nodes = [
        spatialTranspose(`${namePrefix}_to_nchw`, inputName, nchwInput, [0, 3, 1, 2]),
        {
            name: `${namePrefix}_conv`,
            opType: 'ConvTranspose',
            inputs: convInputs,
            outputs: [nchwOutput],
            attributes: [
                { name: 'kernel_shape', type: 'INTS', value: [BigInt(kernelHeight), BigInt(kernelWidth)] },
                { name: 'strides', type: 'INTS', value: [BigInt(stride), BigInt(stride)] },
                { name: 'auto_pad', type: 'STRING', value: padding === 'same' ? 'SAME_UPPER' : 'VALID' },
            ],
        },
        spatialTranspose(`${namePrefix}_to_nhwc`, nchwOutput, nhwcOutput, [0, 2, 3, 1]),
    ];

    const activationName = layer.activation_function ? layer.activation_function.name : 'linear';
    const activationOp = ACTIVATION_TO_ONNX_OP[activationName];
    if (activationOp === undefined) {
        throw new Error(`translateTransConv: activation "${activationName}" (layer index ${layerIndex}) has no ONNX equivalent registered in ACTIVATION_TO_ONNX_OP`);
    }
    let outputName = nhwcOutput;
    if (activationOp !== null) {
        outputName = `${namePrefix}_activation_out`;
        nodes.push({
            name: `${namePrefix}_${activationName}`,
            opType: activationOp,
            inputs: [nhwcOutput],
            outputs: [outputName],
        });
    }

    return { nodes, initializers, outputName };
}

module.exports = {
    translateConnectedLayer,
    translateReshape,
    translateLayerNorm,
    translateMaxPool,
    translateConvLayer,
    translateTransConv,
    ACTIVATION_TO_ONNX_OP,
};
const { green, reset, red } = require('../../../color-code')
const fs = require('fs');
const path = require('path');
const { translateConnectedLayer, translateReshape, translateLayerNorm, translateConvLayer, translateMaxPool } = require('./translators');

const SUPPORTED_LAYER_TYPES = new Set(['Connected Layer', 'Reshape', 'Layer Normalization', "Convolutional Layer", "Max Pooling"]);

// Maps a layer's layer_name to the translate* function that knows how to
// turn it into ONNX node/initializer descriptors. Every entry here must
// also be listed in SUPPORTED_LAYER_TYPES above.
const LAYER_TRANSLATORS = {
    'Connected Layer': translateConnectedLayer,
    'Reshape': translateReshape,
    'Layer Normalization': translateLayerNorm,
    'Convolutional Layer': translateConvLayer,
    'Max Pooling': translateMaxPool
};

/**
 * Converts a TypedArray's underlying memory into a Uint8Array view
 * (zero-copy) suitable for TensorProto.rawData. Works for Float32Array
 * (weights/biases/gamma/beta) as well as BigInt64Array (e.g. Reshape's
 * int64 shape initializer).
 * @param {Float32Array|BigInt64Array} arr
 * @returns {Uint8Array}
 */
const toRawData = (arr) => new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);

/**
 * @param {String} filename output filename (without extension - .onnx is appended)
 * @param {Array<Object>} layers  Neurex's this.layers - must all be "Connected Layer"
 * @param {Array<Float32Array>} weights  Neurex's this.weights
 * @param {Array<Float32Array>} biases   Neurex's this.biases
 */
const exportToOnnx = async (filename, layers, weights, biases) => {
    if (!layers || layers.length === 0) {
        throw new Error('exportToOnnx: no layers to export');
    }

    const unsupported = layers.find((l) => !SUPPORTED_LAYER_TYPES.has(l.layer_name));
    if (unsupported) {
        throw new Error(`exportToOnnx: layer type "${unsupported.layer_name}" is not supported yet. Only ${[...SUPPORTED_LAYER_TYPES].join(', ')} can be exported to ONNX right now.`);
    }
    
    const { create, toBinary } = await import('@bufbuild/protobuf');
    const {
        ModelProtoSchema,
        GraphProtoSchema,
        NodeProtoSchema,
        TensorProtoSchema,
        ValueInfoProtoSchema,
        TypeProtoSchema,
        TypeProto_TensorSchema,
        TensorShapeProtoSchema,
        TensorShapeProto_DimensionSchema,
        TensorProto_DataType,
        OperatorSetIdProtoSchema,
        AttributeProtoSchema,
        AttributeProto_AttributeType,
    } = await import('onnx-buf');

    const makeValueInfo = (name, dims) =>
        create(ValueInfoProtoSchema, {
            name,
            type: create(TypeProtoSchema, {
                value: {
                    case: 'tensorType',
                    value: create(TypeProto_TensorSchema, {
                        elemType: TensorProto_DataType.FLOAT,
                        shape: create(TensorShapeProtoSchema, {
                            dim: dims.map((d) =>
                                create(TensorShapeProto_DimensionSchema, {
                                    value: { case: 'dimValue', value: BigInt(d) },
                                })
                            ),
                        }),
                    }),
                },
            }),
        });

    // translate* functions default to float32 initializers (weights/biases/
    // gamma/beta), but some (e.g. Reshape's shape tensor) need a different
    // ONNX dtype - they flag this via an explicit `dataType` string on the
    // initializer descriptor (e.g. 'INT64').
    const makeTensor = (name, dims, data, dataType) => {
        const onnxDataType = dataType ? TensorProto_DataType[dataType] : TensorProto_DataType.FLOAT;
        if (onnxDataType === undefined) {
            throw new Error(`makeTensor: unknown ONNX dataType "${dataType}" for initializer "${name}"`);
        }

        return create(TensorProtoSchema, {
            name,
            dims: dims.map((d) => BigInt(d)),
            dataType: onnxDataType,
            rawData: toRawData(data),
        });
    };

    const ATTRIBUTE_VALUE_FIELDS = {
        FLOAT: 'f',
        INT: 'i',
        STRING: 's',
        FLOATS: 'floats',
        INTS: 'ints',
        STRINGS: 'strings',
    };

    const makeAttribute = (attrDescriptor) => {
        const { name, type, value } = attrDescriptor;
        const onnxType = AttributeProto_AttributeType[type];
        const field = ATTRIBUTE_VALUE_FIELDS[type];

        if (onnxType === undefined || !field) {
            throw new Error(`makeAttribute: unsupported attribute type "${type}" for attribute "${name}"`);
        }

        return create(AttributeProtoSchema, {
            name,
            type: onnxType,
            [field]: type === 'STRING' ? new TextEncoder().encode(value) : value,
        });
    };

    const makeNode = (nodeDescriptor) =>
        create(NodeProtoSchema, {
            name: nodeDescriptor.name,
            opType: nodeDescriptor.opType,
            input: nodeDescriptor.inputs,
            output: nodeDescriptor.outputs,
            attribute: (nodeDescriptor.attributes || []).map(makeAttribute),
        });

    const allNodes = [];
    const allInitializers = [];

    // Graph-level input/output are modeled flat as [1, N] (batch of 1, N
    // features) regardless of layer type, matching makeValueInfo's usage
    // below. For Connected Layer this N is weightShape[0]/[1]; for
    // shape-preserving/shape-changing layers (LayerNorm, Reshape) it's the
    // product of inputShape/outputShape, since those can be multi-dim
    // (e.g. Reshape's targetShape [28, 28, 3]).
    const flatSize = (layer, shapeKey) => {
        const shape = layer[shapeKey];
        if (!shape || shape.length === 0) {
            throw new Error(`exportToOnnx: layer "${layer.layer_name}" is missing a valid ${shapeKey} (needed to determine graph ${shapeKey === 'inputShape' ? 'input' : 'output'} size) ([${shape}])`);
        }
        return shape.reduce((acc, d) => acc * d, 1);
    };

    const firstLayer = layers[0];
    const firstInputSize = firstLayer.layer_name === 'Connected Layer'
        ? firstLayer.weightShape[0]
        : flatSize(firstLayer, 'inputShape');
    const graphInputShape = firstLayer.inputShape && firstLayer.inputShape.length > 1
        ? [1, ...firstLayer.inputShape]
        : [1, firstInputSize];
    let currentInputName = 'input';
    
    let pointer = 0;
    layers.forEach((layer, layerIndex) => {
        const translate = LAYER_TRANSLATORS[layer.layer_name];

        if (weights[pointer].some(n => isNaN(n))) {
            console.log(`${red}[ERROR]${reset} Parameter of Layer ${layer.layer_name} ${layerIndex} has NaNs`);
            throw new Error("ERR_PARAM_HAS_NAN");
        }

        const { nodes, initializers, outputName } = translate(layer, weights[pointer], biases[pointer], currentInputName, layerIndex);

        nodes.forEach((n) => allNodes.push(makeNode(n)));
        initializers.forEach((t) => allInitializers.push(makeTensor(t.name, t.dims, t.data, t.dataType)));

        currentInputName = outputName;

        if (layer.isParametric) {
            pointer++;
        }
    });

    const lastLayer = layers[layers.length - 1];
    const graphOutputShape = lastLayer.outputShape && lastLayer.outputShape.length > 1
        ? [1, ...lastLayer.outputShape]
        : [1, lastLayer.weightShape ? lastLayer.weightShape[1] : flatSize(lastLayer, 'outputShape')];

    const graph = create(GraphProtoSchema, {
        name: filename,
        node: allNodes,
        initializer: allInitializers,
        input: [makeValueInfo('input', graphInputShape)],
        output: [makeValueInfo(currentInputName, graphOutputShape)],
    });

    const model = create(ModelProtoSchema, {
        irVersion: BigInt(10),
        producerName: 'neurex',
        graph,
        opsetImport: [create(OperatorSetIdProtoSchema, { domain: '', version: BigInt(21) })],
    });

    const bytes = toBinary(ModelProtoSchema, model);
    const outputPath = path.join(process.cwd(), `${filename}.onnx`);
    fs.writeFileSync(outputPath, bytes);

    console.log(`${green}[SUCCESS]${reset} Model ${filename}.onnx has been saved`)
};

module.exports = exportToOnnx;
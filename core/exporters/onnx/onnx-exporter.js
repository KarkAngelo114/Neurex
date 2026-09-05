const { green, reset } = require('../../../color-code')
const fs = require('fs');
const path = require('path');
const { translateConnectedLayer } = require('./translators');

const SUPPORTED_LAYER_TYPES = new Set(['Connected Layer']);

/**
 * Converts a Float32Array's underlying memory into a Uint8Array view
 * (zero-copy) suitable for TensorProto.rawData.
 * @param {Float32Array} arr
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

    const makeTensor = (name, dims, data) =>
        create(TensorProtoSchema, {
            name,
            dims: dims.map((d) => BigInt(d)),
            dataType: TensorProto_DataType.FLOAT,
            rawData: toRawData(data),
        });

    const makeNode = (nodeDescriptor) =>
        create(NodeProtoSchema, {
            name: nodeDescriptor.name,
            opType: nodeDescriptor.opType,
            input: nodeDescriptor.inputs,
            output: nodeDescriptor.outputs,
        });

    const allNodes = [];
    const allInitializers = [];

    const firstLayer = layers[0];
    const [firstInputSize] = firstLayer.weightShape;
    let currentInputName = 'input';

    layers.forEach((layer, layerIndex) => {
        const { nodes, initializers, outputName } = translateConnectedLayer(layer, weights[layerIndex], biases[layerIndex], currentInputName, layerIndex);

        nodes.forEach((n) => allNodes.push(makeNode(n)));
        initializers.forEach((t) => allInitializers.push(makeTensor(t.name, t.dims, t.data)));

        currentInputName = outputName;
    });

    const lastLayer = layers[layers.length - 1];
    const [, lastOutputSize] = lastLayer.weightShape;

    const graph = create(GraphProtoSchema, {
        name: filename,
        node: allNodes,
        initializer: allInitializers,
        input: [makeValueInfo('input', [1, firstInputSize])],
        output: [makeValueInfo(currentInputName, [1, lastOutputSize])],
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
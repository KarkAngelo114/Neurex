const fs = require('fs');
const { reset, lime } = require('../../../color-code');
const onnx = require('onnx-proto').onnx;

// ---------------------------------------------------------------------------
// Activation name -> ONNX op_type lookup.
// NOTE: these are the common ones. Cross-check the exact lowercase strings
// your activation_functions module actually produces (activation_function.name)
// against this table — add/rename entries as needed.
// ---------------------------------------------------------------------------
const ACTIVATION_TO_ONNX_OP = {
    relu: 'Relu',
    sigmoid: 'Sigmoid',
    tanh: 'Tanh',
    softmax: 'Softmax',
    leaky_relu: 'LeakyRelu',
    elu: 'Elu',
    linear: null,
    none: null,
};

/**
 * Builds a TensorProto (FLOAT) wrapping a Float32Array's raw bytes.
 * Zero-copy where possible - reuses the same underlying-buffer trick
 * the .nrx save format already relies on.
 *
 * @param {string} name
 * @param {Float32Array} arr
 * @param {number[]} dims
 */
function makeFloatTensor(name, arr, dims) {
    const buf = Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);

    return onnx.TensorProto.create({
        name,
        dims,
        dataType: onnx.TensorProto.DataType.FLOAT,
        rawData: buf,
    });
}

function makeValueInfo(name, dims) {
    return onnx.ValueInfoProto.create({
        name,
        type: onnx.TypeProto.create({
            tensorType: onnx.TypeProto.Tensor.create({
                elemType: onnx.TensorProto.DataType.FLOAT,
                shape: onnx.TensorShapeProto.create({
                    dim: dims.map(d => onnx.TensorShapeProto.Dimension.create({ dimValue: d })),
                }),
            }),
        }),
    });
}

/**
 * @param {String} filename output filename (without extension - .onnx is appended)
 * @param {Array<Object>} layers  Neurex's this.layers - must all be "Connected Layer"
 * @param {Array<Float32Array>} weights  Neurex's this.weights
 * @param {Array<Float32Array>} biases   Neurex's this.biases
 * @param {Object} modelMeta  { input_size, input_shape, output_size }
 */
const exportToOnnx = async (filename, layers, weights, biases, modelMeta = {}) => {
    // ---- 1. Validate: ANN-only support for now ----------------------------
    const unsupported = layers.filter(l => l.layer_name !== 'Connected Layer');

    if (unsupported.length > 0) {
        const names = [...new Set(unsupported.map(l => l.layer_name))].join(', ');
        throw new Error(
            `[ONNX Export] Only "Connected Layer" models are currently supported. ` +
            `Found unsupported layer type(s): ${names}. ` +
            `Conv/Pooling/Attention/Recurrent export support is planned but not yet implemented.`
        );
    }

    const nodes = [];
    const initializers = [];

    // input_size may be undefined on some load paths - fall back to weightShape.
    const inputSize = modelMeta.input_size
        || (layers[0] && layers[0].weightShape && layers[0].weightShape[0]) // weightShape = [inputSize, outputSize]
        || (layers[0] && layers[0].inputShape && layers[0].inputShape.reduce((a, b) => a * b, 1));

    let currentTensorName = 'input';

    layers.forEach((layer, i) => {
        const w = weights[i];
        const b = biases[i];

        // weightShape is stored as [inputSize, outputSize] per Neurex's connectedLayer.js
        // initParams(): `const weightShape = [inputSize, outputSize];`
        // The flat weight buffer is laid out to match - feedforward() calls
        // MatMul(input, inputSize, outputSize, ...), meaning the buffer is
        // row-major as [inputSize rows][outputSize cols] (one row per input feature).
        //
        // ONNX Gemm(transB=1) expects B as [outFeatures, inFeatures] row-major.
        // That's the TRANSPOSE of Neurex's native layout - so we transpose the
        // actual float data here, not just relabel the shape.
        const [inFeatures, outFeatures] = layer.weightShape;

        const wName = `W${i}`;
        const bName = `B${i}`;

        const wTransposed = new Float32Array(inFeatures * outFeatures);
        for (let r = 0; r < inFeatures; r++) {
            for (let c = 0; c < outFeatures; c++) {
                // source: row-major [inFeatures, outFeatures] -> w[r * outFeatures + c]
                // dest:   row-major [outFeatures, inFeatures] -> wT[c * inFeatures + r]
                wTransposed[c * inFeatures + r] = w[r * outFeatures + c];
            }
        }

        initializers.push(makeFloatTensor(wName, wTransposed, [outFeatures, inFeatures]));

        const gemmOutputName = `gemm_out_${i}`;
        const gemmInputs = [currentTensorName, wName];

        if (layer.useBias && b) {
            initializers.push(makeFloatTensor(bName, b, [outFeatures]));
            gemmInputs.push(bName);
        }

        nodes.push(onnx.NodeProto.create({
            opType: 'Gemm',
            input: gemmInputs,
            output: [gemmOutputName],
            name: `Gemm_${i}`,
            attribute: [
                onnx.AttributeProto.create({ name: 'alpha', f: 1.0, type: onnx.AttributeProto.AttributeType.FLOAT }),
                onnx.AttributeProto.create({ name: 'beta', f: 1.0, type: onnx.AttributeProto.AttributeType.FLOAT }),
                onnx.AttributeProto.create({ name: 'transB', i: 1, type: onnx.AttributeProto.AttributeType.INT }),
            ],
        }));

        currentTensorName = gemmOutputName;

        // ---- activation node (if any) --------------------------------------
        const actName = layer.activation_function && layer.activation_function.name
            ? layer.activation_function.name.toLowerCase()
            : null;

        if (actName && ACTIVATION_TO_ONNX_OP[actName] === undefined) {
            console.warn(
                `[ONNX Export] WARNING: unrecognized activation "${actName}" on layer ${i}. ` +
                `Skipping activation node - check ACTIVATION_TO_ONNX_OP mapping in onnx-exporter.js.`
            );
        } else if (actName && ACTIVATION_TO_ONNX_OP[actName]) {
            const activationOutputName = `act_out_${i}`;
            nodes.push(onnx.NodeProto.create({
                opType: ACTIVATION_TO_ONNX_OP[actName],
                input: [currentTensorName],
                output: [activationOutputName],
                name: `${ACTIVATION_TO_ONNX_OP[actName]}_${i}`,
            }));
            currentTensorName = activationOutputName;
        }
        // if actName maps to null (linear/none) or is absent, no node - passthrough.
    });

    // ---- 2. Rename the final tensor to "output" so graph outputs line up ----
    if (nodes.length > 0) {
        const lastNode = nodes[nodes.length - 1];
        lastNode.output[0] = 'output';
    }

    const lastLayer = layers[layers.length - 1];
    const outputSize = modelMeta.output_size || lastLayer.weightShape[1]; // weightShape = [inputSize, outputSize]

    const graph = onnx.GraphProto.create({
        name: 'neurex_graph',
        node: nodes,
        initializer: initializers,
        input: [makeValueInfo('input', [1, inputSize])],
        output: [makeValueInfo('output', [1, outputSize])],
    });

    const model = onnx.ModelProto.create({
        irVersion: 8,
        producerName: 'neurex',
        opsetImport: [onnx.OperatorSetIdProto.create({ version: 13 })],
        graph,
    });

    const errMsg = onnx.ModelProto.verify(model);
    if (errMsg) {
        throw new Error(`[ONNX Export] Built an invalid ModelProto: ${errMsg}`);
    }

    const buffer = onnx.ModelProto.encode(model).finish();
    const outPath = `${filename}.onnx`;
    fs.writeFileSync(outPath, buffer);

    console.log(`${lime}[SUCCESS]${reset} Model exported to ${outPath}`);
    return outPath;
};

module.exports = exportToOnnx;

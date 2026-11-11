const Layers = require('../layers');
const color = require('../color-code');
const errFlags = require('../err_flags');
const UUID = require('crypto');
const { createTensorBuffer } = require('../utils/utils');

/**
 * Use to parse the model adn restructure it back
 * @param {Array<Object>} model_layers 
 * @param {Array<Float32Array>} weights 
 * @param {Array<Float32Array>} biases 
 * @param {Boolean} freezeWeights 
 * @returns {Array<Object>}
 */
const parseModel= (model_layers, weights, biases, freezeWeights) => {

    const layerBuilder = new Layers();
    let paramPointer = 0;

    const layers = model_layers.map(layerData => {


        let newLayer;
        if (layerData.layer_name === "Connected Layer") {
            newLayer = layerBuilder.connectedLayer(layerData.layer_size, layerData.activation_function_name, layerData.useBias);
            newLayer.weightShape = layerData.weightShape;
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Convolutional Layer") {
            newLayer = layerBuilder.convolutionalLayer(layerData.filters, layerData.strides, layerData.kernel_size, layerData.activation_function_name, layerData.padding, layerData.useBias);
            newLayer.weightShape = layerData.weightShape;
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.layerID = UUID.randomUUID();
        } 
        else if (layerData.layer_name === "Max Pooling") {
            newLayer = layerBuilder.maxPooling(layerData.poolSize, layerData.strides, layerData.padding);
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Embedding Layer") {
            const vocabSize = layerData.vocabSize;
            const embeddingDim = layerData.embeddingDim;
            const sequence_length = layerData.maxSequenceLength;
            const outputSize = sequence_length * embeddingDim;
            newLayer = layerBuilder.embeddingLayer(vocabSize, embeddingDim, sequence_length);
            newLayer.inputShape = [1, 1, sequence_length];
            newLayer.outputShape = [1, 1, embeddingDim, sequence_length];
            newLayer.weightShape = [vocabSize, embeddingDim];
            newLayer.outputSize = outputSize;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Recurrent Cell") {
            const units = layerData.units;
            const return_sequence = layerData.return_sequence || false;
            newLayer = layerBuilder.recurrentCell(units, layerData.activation_function_name, return_sequence, layerData.useBias);
            newLayer.weightShape = layerData.weightShape;
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.seqLen = layerData.maxSequenceLength || 1;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Trans Convolution") {
            const filters = layerData.filters;
            const strides = layerData.strides;
            const kernelSize = layerData.kernel_size
            newLayer = layerBuilder.transConvLayer(filters, strides, kernelSize, layerData.activation_function_name, layerData.padding, layerData.useBias);
            newLayer.weightShape = layerData.weightShape;
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Reshape") {
            newLayer = layerBuilder.reshape(layerData.targetShape);
            newLayer.weightShape = layerData.weightShape;
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Simple Attention") {
            newLayer = layerBuilder.simpleAttention(layerData.useCausalMasking, layerData.useBias);
            newLayer.weightShape = layerData.weightShape;
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.dkRoot = layerData.dkRoot;
            newLayer.embedDim = layerData.embeddingDim;
            newLayer.seqLen = layerData.maxSequenceLength;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Multi Head Attention") {
            newLayer = layerBuilder.multiHeadAttention(layerData.numHeads, layerData.useCausalMasking, layerData.useBias);
            newLayer.weightShape = layerData.weightShape;
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.dkRoot = layerData.dkRoot;
            newLayer.embedDim = layerData.embeddingDim;
            newLayer.seqLen = layerData.maxSequenceLength;
            newLayer.numHeads = layerData.numHeads;
            newLayer.headDim = layerData.headDim;
            newLayer.useCausalMasking = layerData.useCausalMasking;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Layer Normalization") {
            newLayer = layerBuilder.layerNorm(layerData.eps);
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.weightShape = layerData.weightShape;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Residual Start") {
            newLayer = layerBuilder.residualStart();
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.weightShape = layerData.weightShape;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Residual End") {
            newLayer = layerBuilder.residualEnd();
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.weightShape = layerData.weightShape;
            newLayer.layerID = UUID.randomUUID();
        }
        else if (layerData.layer_name === "Sinusoidal Encoding") {
            newLayer = layerBuilder.sinusoidalEncoding();
            newLayer.inputShape = layerData.inputShape;
            newLayer.outputShape = layerData.outputShape;
            newLayer.embeddingDim = layerData.embeddingDim;
            newLayer.maxSequenceLength = layerData.maxSequenceLength;
            newLayer.layerID = UUID.randomUUID();
        }
        else {
            console.error(`${color.red}[ERROR]${color.reset} Unknown layer type '${layerData.layer_name}' found in model. It might you've loaded a model that is not compatible to this version of Neurex yet nor has the corresponding layer type to map back. To confirm, you can check layer definitions in ${color.gray}https://neurex-documentation.vercel.app/javascript-nodejs#layers${color.reset} or update to the newest version of Neurex.`);
            throw new Error(errFlags.PARSE_ERROR);
        }

        if (layerData.isParametric) {  
            newLayer.weights = weights[paramPointer];
            newLayer.biases = biases[paramPointer] || createTensorBuffer([1], {prefilledWith:"zeroes"}); // dummy bias buffer because the optimizers needs boht buffers, but won't get updated.
            newLayer.freezeWeights = freezeWeights;
            paramPointer++;
        }
        
        return newLayer;
    });

    return layers;
}

module.exports = {
    parseModel
};
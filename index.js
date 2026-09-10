/**
 * 

 Centralized imports to tbe main script. 

 Example Usage:
 const {Neurex, MinMaxScaler, ...} = require('neurex');

 */

const Neurex = require('./core');
const CsvDataHandler = require('./preprocessor/CsvDataHandler');
const {MinMaxScaler} = require('./preprocessor/normalizer');
const Layers = require('./layers');
const {OneHotEncoded, IntegerLabeling, BinaryLabeling} = require('./preprocessor/label_encoder');
const split_dataset = require('./preprocessor/split');
const RegressionMetrics = require('./metrics/regression_metrics');
const ClassificationMetrics = require('./metrics/classification_metrics');
const { load_images_from_directory, load_single_image, load_multiple_images } = require('./preprocessor/imagery');
const { element_wise_mul, relu, sigmoid, tanh, softmax, linear, element_wise_sub, scaleDiff  } = require('./core/bindings/float32Ops');
const { Annotator } = require('./preprocessor/annotator');
const { detectGPU } = require('./gpu/gpu_init');
const { simpleNeuralNetwork, simpleCNN, vanillaRNN } = require('./applications/templates');
const { Encode, buildVocab, buildWord2Id, tokenize } = require('./preprocessor/tokenizer');
const { stepDecay, exponentialDecay, cosineAnnealing, reduceOnPlateau } = require('./schedulers');
const { SGD, Adam, RMSprop } = require('./optimizers');
const { lossVisualizer } = require('./applications/visualizer/lossVisualizer');
const { lossLandscapeVisualizer } = require('./applications/visualizer/lossLandscapeVisualizer');
const { modelVisualizer } = require('./applications/visualizer/modelVisualizer');
const { clipGradient } = require('./normalizers');



module.exports= {
    Neurex,
    CsvDataHandler,
    MinMaxScaler,
    Annotator,
    Layers,
    detectGPU,
    lossVisualizer,
    lossLandscapeVisualizer,
    modelVisualizer,
    templates: {
        simpleNeuralNetwork,
        simpleCNN,
        vanillaRNN,
    },
    gradientNormalizers: {
        clipGradient
    },
    optimizers: {
        SGD,
        Adam,
        RMSprop,
    },
    schedulers: {
        stepDecay,
        exponentialDecay,
        cosineAnnealing,
        reduceOnPlateau,
    },
    metrics: {
        RegressionMetrics,
        ClassificationMetrics,
    },
    preprocesors: {
        OneHotEncoded,
        IntegerLabeling,
        BinaryLabeling,
        split_dataset,
        load_images_from_directory,
        load_single_image,
        load_multiple_images,
        tokenize,
        buildWord2Id,
        buildVocab,
        Encode,
    },
    math: {
        element_wise_mul,
        element_wise_sub,
        scaleDiff,
        relu,
        sigmoid,
        tanh,
        softmax,
        linear,
    }
}
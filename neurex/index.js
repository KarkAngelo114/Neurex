/**
 * 

 Centralized imports to tbe main script. 

 Example Usage:
 const {Neurex, Interpreter, MinMaxScaler, ...} = require('neurex');

 */

const Neurex = require('./core');
const Interpreter = require('./interpreter');
const CsvDataHandler = require('./preprocessor/CsvDataHandler');
const {MinMaxScaler} = require('./preprocessor/normalizer');
const {OneHotEncoded, IntegerLabeling, BinaryLabeling} = require('./preprocessor/label_encoder');
const split_dataset = require('./preprocessor/split');
const RegressionMetrics = require('./metrics/regression_metrics');
const ClassificationMetrics = require('./metrics/classification_metrics');

module.exports= {
    Neurex,
    Interpreter,
    CsvDataHandler,
    MinMaxScaler,
    OneHotEncoded,
    IntegerLabeling,
    BinaryLabeling,
    split_dataset,
    RegressionMetrics,
    ClassificationMetrics
}
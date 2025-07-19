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
const split_dataset = require('./preprocessor/split');
const RegressionMetrics = require('./metrics/regression_metrics');

module.exports= {
    Neurex,
    Interpreter,
    CsvDataHandler,
    MinMaxScaler,
    split_dataset,
    RegressionMetrics
}
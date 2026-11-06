
const { red, reset } = require('../color-code');
const errFlags = require('../err_flags');
const { ApplySGD, ApplyAdam, ApplyRMSProp } = require('../core/bindings');
const { createTensorBuffer } = require('../utils/utils');


module.exports = {
    SGD: (momentum = 0.9) => {
        // Name the inner function sgd
        return function SGD(data) {
            const {
                weights,
                biases,
                learning_rate,
                batchSize = 1,
                pointer,
                modelID,
                layerData,
                layerID
            } = data;

            const weightGrads = layerData.weightGrads;
            const biasGrads = layerData.biasGrads;
            const useBias = layerData.useBias;

            const weight_velocity = layerData.weights_velocity || createTensorBuffer([weights.length], {prefilledWith:"zeroes"}).data;
            const bias_velocity = layerData.bias_velocity || createTensorBuffer([biases.length], {prefilledWith:"zeroes"}).data;
            
            if (weights.length !== weightGrads.length ||
                (useBias && (!biases || !biasGrads || biases.length !== biasGrads.length))) {
                console.error("SGD: Params and grads size mismatch");
                throw new Error(errFlags.UNEQUAL_SIZE);
            }

            const result = ApplySGD(
                weights,
                weightGrads,
                weight_velocity,
                biases,
                biasGrads,
                bias_velocity,
                learning_rate,
                momentum,
                batchSize,
                useBias,
                pointer,
                modelID,
                layerID
            );

            
            
            layerData.weights_velocity = result.weightVelocity;
            if (useBias) layerData.bias_velocity = result.biasVelocity;
            

            return {
                weights: result.weights,
                biases: result.biases,
            };
        };
    },

    Adam: (beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) => {
        // Name the inner function adam
        return function Adam(data) {

            const {
                weights,
                biases,
                learning_rate,
                batchSize = 1,
                pointer,
                modelID,
                layerData,
                layerID
            } = data;

            const weightGrads = layerData.weightGrads;
            const biasGrads = layerData.biasGrads;
            const useBias = layerData.useBias;

            if (weights.length !== weightGrads.length ||
                (useBias && (!biases || !biasGrads || biases.length !== biasGrads.length))) {
                console.error("SGD: Params and grads size mismatch");
                throw new Error(errFlags.UNEQUAL_SIZE);
            }

            const state_M_weights = layerData.state_M_weights || createTensorBuffer([weights.length], {prefilledWith:"zeroes"}).data;
            const state_V_weights = layerData.state_V_weights || createTensorBuffer([weights.length], {prefilledWith:"zeroes"}).data;
            const state_M_bias = layerData.state_M_bias || createTensorBuffer([biases.length], {prefilledWith:"zeroes"}).data;
            const state_V_bias = layerData.state_V_bias || createTensorBuffer([biases.length], {prefilledWith:"zeroes"}).data;

            let state_t = layerData.state_t || 0;
            state_t += 1;

            const res = ApplyAdam(weights, weightGrads, biases, biasGrads, state_M_weights, state_M_bias, state_V_weights, state_V_bias, state_t, learning_rate, beta1, beta2, epsilon, batchSize, useBias, modelID, layerID, pointer);
            layerData.state_M_weights = res.m_weights;
            layerData.state_V_weights = res.v_weights;
            layerData.state_M_bias = res.m_biases;
            layerData.state_V_bias = res.v_biases;

            return {
                weights: res.weights,
                biases: res.biases,
            };
        };
    },

    RMSprop: (decayRate = 0.9, epsilon = 1e-8) => {
        
        return function RMSprop(data) {
            const {
                weights,
                biases,
                learning_rate,
                batchSize = 1,
                pointer,
                modelID,
                layerData,
                layerID
            } = data;

            const weightGrads = layerData.weightGrads;
            const biasGrads = layerData.biasGrads;
            const useBias = layerData.useBias;

            if (weights.length !== weightGrads.length ||
                (useBias && (!biases || !biasGrads || biases.length !== biasGrads.length))) {
                console.error("SGD: Params and grads size mismatch");
                throw new Error(errFlags.UNEQUAL_SIZE);
            }

            const sqAvg_weights = layerData.sqAvg_weights || createTensorBuffer([weights.length], {prefilledWith:"zeroes"}).data;
            const sqAvg_biases = layerData.sqAvg_biases || createTensorBuffer([biases.length], {prefilledWith:"zeroes"}).data;


            const res = ApplyRMSProp(weights, weightGrads, biases, biasGrads, sqAvg_weights, sqAvg_biases, learning_rate, epsilon, decayRate, batchSize, useBias, modelID, layerID, pointer);
            layerData.sqAvg_weights = res.sqAvg_weights;
            layerData.sqAvg_biases = res.sqAvg_biases;
            
            return {
                weights: res.weights,
                biases: res.biases,
            }
        }
    }
};
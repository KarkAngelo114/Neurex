
/**
 * 

 This interpreter is dedicated for Neurex

 */


const activation = require('../gpu/kernels/activations');
const detect = require('../gpu/detectGPU');
const { computeForward } = require('../gpu/kernels/matrixMultiplication');
const zlib = require('zlib');
const fs = require('fs');
const path = require('path');



/**
 * This class allows you to run inference predictions on your applications. You can load your trained model and run predictions
 *
 * @class
 */
class Interpreter {
    constructor () {
        this.weights = []; // weights for calculating the dot product of each nueron
        this.biases = []; // biases for calculating the dot product of each nueron
        this.num_layers = 0; // the number layers in a network (hidden and the output layer) 
        this.layers = [];
        this.input_size = 0; // the size of the input layer, basically the number of input neurons.
        this.output_size = 0;
        this.onGPU = true;
    }

    /**
    * @method loadSavedModel()
    * @param {*} model - the trained model

    The loadSavedModel() method allows you to load the trained model. The model is typically in .nrx file format which contains the learned parameters of your trained model

    */
    loadSavedModel(model) {
        try {
            if (!model) {
                throw new Error("No model provided");
            }

            const dir = path.dirname(require.main.filename);
            const model_file = path.join(dir, `${model}`);

            // Check extension
            if (path.extname(model_file) !== '.nrx') {
                throw new Error("Invalid file type. Only .nrx model files are supported.");
            }

            // Read file
            const rawBuffer = fs.readFileSync(model_file);

            // Validate magic header
            const header = rawBuffer.slice(0, 4).toString('utf-8');
            if (header !== 'NRX2') {
                throw new Error("Invalid file format. Not a valid NRX model.");
            }

            // Check version
            const version = rawBuffer[4];
            if (version !== 0x02) {
                throw new Error(`Unsupported NRX version: ${version}`);
            }

            // Decompress and parse
            const compressedData = rawBuffer.slice(5);
            const jsonString = zlib.inflateSync(compressedData).toString('utf-8');
            const modelData = JSON.parse(jsonString);

            // Assign properties
            this.task = modelData.task;
            this.loss_function = modelData.loss_function;
            this.epoch_count = modelData.epoch;
            this.batch_size = modelData.batch_size;
            this.optimizer = modelData.optimizer;
            this.learning_rate = modelData.learning_rate;
            this.input_size = modelData.input_size;
            this.output_size = modelData.output_size;
            this.num_layers = modelData.num_layers;
            this.weights = modelData.weights;
            this.biases = modelData.biases;
            this.layers = modelData.layers.map(layerData => {
                const layer = {
                    layer_name: layerData.layer_name,
                    layer_size: layerData.layer_size
                };

                if (layerData.activation_function_name) {
                    const funcName = layerData.activation_function_name.toLowerCase();
                    if (!activation[funcName]) {
                        throw new Error(`[ERROR] Unknown activation function '${funcName}' found in model.`);
                    }
                    layer.activation_function = activation[funcName];
                }

                if (layerData.derivative_activation_function_name) {
                    let derivFuncName = layerData.derivative_activation_function_name; 

                    let keyInDerivatives = derivFuncName.startsWith('d') ? derivFuncName.substring(1).toLowerCase(): derivFuncName.toLowerCase();

                    if (!activation.derivatives || !activation.derivatives[keyInDerivatives]) {
                        throw new Error(`[ERROR] Unknown derivative activation function '${layerData.derivative_activation_function_name}' found in model.`);
                    }
                    layer.derivative_activation_function = activation.derivatives[keyInDerivatives];
                }
                return layer;
            });

            console.log(`[SUCCESS]------- Model ${model} successfully loaded`);
        } catch (error) {
            console.log(error.message);
        }
    }

    /**
     * 

    @method modelSummary()

    Shows the model architecture
     */
    modelSummary() {
        console.log(`Input size: ${this.input_size}`);
        console.log(`Number of layers: ${this.num_layers}`);
        console.log('Layer\tNeurons\tActivation');
        for (let i = 0; i < this.num_layers; i++) {
            const actFn = this.activation_functions[i];
            // Try to get the function name, fallback to 'custom' if not available
            const actName = actFn && actFn.name ? actFn.name : (typeof actFn === 'string' ? actFn : 'custom');
            console.log(`${i + 1}\t${this.number_of_neurons[i]}\t${actName}`);
        }
    }

    /**
     * 
     @method predict()
     @param {Array} input - input data 
     @returns Array of predictions
     @throws Error when there's shape mismatch and no input data

     produces predictions based on the input data
     */
    predict(input) {
        try {
            if (!input) {
                throw new Error("\n[ERROR]-------No inputs")
            }

            if (input[0].length != this.input_size) {
                throw new Error(`\n[ERROR]-------Shape Mismatch | Input shape length: ${input[0].length} | Expecting ${this.input_size}`);
            }

            const {gpu, backend, isGPUAvailable, isSoftwareGPU} = detect();

            if (!isGPUAvailable || isSoftwareGPU) {
                console.log(`[INFO]------- Falling back to CPU mode (no GPU acceleration)`);
                this.onGPU = false;
            } else {
                console.log(`[INFO]-------- Backend Detected: ${backend}. Using ${gpu}`);
                this.onGPU = true;
            }

            let outputs = [];
            for (let sample_index = 0; sample_index < input.length; sample_index++) {
                /**
                 * 
                 we loop through the entire loaded dataset
                 */
                const array_of_features = input[sample_index];
                // perform feedforward, similar when training but only outputs predictions. No updating of weights and biases.
                const {predictions} = this.#Feedforward(array_of_features);
                outputs.push(predictions);
            }
            return outputs;
        }
        catch (err) {
            console.error(err.message);
        }
    }

    // forward propagation
    #Feedforward(input) {
        let current_input = input
        let all_layer_outputs = [input];
        let zs = [];

        for (let layer = 0; layer < this.num_layers; layer++) {
            const current_layer = this.layers[layer];
            // all operations inside a connected layer
            if (current_layer.layer_name === "connected_layer") {
                const layer_biases = this.biases[layer];
                const layer_weights = this.weights[layer];
                //const num_neurons = this.number_of_neurons[layer]; the number of biases in a layer can be use to determine how many neurons are there in a connected layer, so this no longer can be use any more
                const activation_function = current_layer.activation_function;

                // compute dot-product for all neurons at once
                const z_values = computeForward(this.onGPU, current_input, layer_weights, layer_biases);

                let outputs;

                // After computing all z_values for the current layer
                if (activation_function.name === "softmax") {
                    outputs = activation_function(z_values); // Apply softmax to all z_values
                } else {
                    // If GPU not available, then perform neuron-by-neuron for getting the activated output
                    if (!this.onGPU) {
                        outputs = [];
                        for (let i= 0; i < layer_biases.length; i++) {
                            outputs.push(activation_function(z_values[i]));
                        }
                    }
                    else {
                        // if GPU available, shove the dot products (z-values or pre-activated outputs) to compute the activated outputs for every neurons
                        outputs = activation_function(z_values, this.onGPU);
                    }
                }
                    
                zs.push(z_values);
                current_input = outputs; // the outputs of the this layer will be the inputs for the next layer (repeat until all the last layer which is the output layer)
                all_layer_outputs.push(current_input); // Push the actual activations
            }
            else {
                // for other layers  . . .
            }
            
        }
        // after all the layers gives off their outputs, return final array of current_input as the predictions
        return {
            predictions: current_input, 
            activations : all_layer_outputs,
            zs: zs
        };
    }
}

module.exports = Interpreter;
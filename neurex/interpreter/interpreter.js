
/**
 * 

 This interpreter is dedicated for Neurex

 */


const Layers = require('../layers');
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
            const layerBuilder = new Layers();
            this.layers = modelData.layers.map(layerData => {
                let newLayer;
                if (layerData.layer_name === "connected_layer") {
                    // Recreate the connected layer with the correct activation and size
                    newLayer = layerBuilder.connectedLayer(layerData.activation_function_name, layerData.layer_size);
                } else if (layerData.layer_name === "input_layer") {
                    // Recreate the input layer. Note: The input layer doesn't have methods, so this is just for consistency
                    newLayer = layerBuilder.inputShape({ features: layerData.layer_size });
                } else if (layerData.layer_name === "flatten_layer") {
                    newLayer = layerBuilder.flatten();
                } else {
                    throw new Error(`[ERROR] Unknown layer type '${layerData.layer_name}' found in model.`);
                }
                
                return newLayer;
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
        console.log("_______________________________________________________________");
        console.log("                        Model Summary                          ");
        console.log("_______________________________________________________________");
        console.log(`Input size: ${this.input_size}`);
        console.log(`Number of layers: ${this.num_layers}`);
        console.log("---------------------------------------------------------------");
        console.log("Layer (type)              Output Shape          Activation     ");
        console.log("===============================================================");
        
        // This will be the shape for the next layer. It starts with the input size.
        let currentOutputShape = this.input_size;

        this.layers.forEach((layer, i) => {
            const layerName = layer.layer_name === "connected_layer" ? "Connected Layer" : layer.layer_name;
            const activationName = layer.activation_function ? layer.activation_function.name : 'None';
            const outputShape = layer.layer_size;

            console.log(
                `Connected Layer           (None, ${outputShape})        ${activationName.padEnd(10, ' ')}`
            );
            
            // Update the output shape for the next iteration
            currentOutputShape = outputShape;
        });
        const total_weights = this.weights.flat(Infinity).length
        const total_biases = this.biases.flat(Infinity).length
        console.log("===============================================================");
        console.log("Total layers: " + this.num_layers);
        console.log("Total Learnable parameters:",parseInt(total_weights+total_biases));
        console.log("===============================================================");
        
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

        for (let layer_index = 0; layer_index < this.num_layers; layer_index++) {
            const current_layer = this.layers[layer_index];
            const layer_weights = this.weights[layer_index];
            const layer_biases = this.biases[layer_index];
            
            // Call the layer's specific feedforward method
            const { outputs, z_values } = current_layer.feedforward(this.onGPU, current_input, layer_weights, layer_biases);

            zs.push(z_values);
            current_input = outputs;
            all_layer_outputs.push(current_input);
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
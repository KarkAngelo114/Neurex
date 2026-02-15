const activation = require('../core/bindings');
const {MatMul, DeltaMatMul, Convolve, StackFeatureMaps, ConvolveDelta} = require('../core/bindings');
const {calculateTensorShape, getPaddingSizes, applyPadding, DilateInput} = require('../utils/utils');
const { toTensor } = require('../preprocessor/reshaper')

class Layers {
    constructor () {
        this.weights = [];
        this.biases = [];
        this.weightGrads = [];
        this.biaeGrads = [];
    }

    /**
     * @method inputShape
     * @param {object} shapeConfig - specify the number of features
     * @example
     * model.sequentialBuild([
        layer.inputShape({features: 4}),
        layer.connectedLayer("relu", 5),
        layer.connectedLayer("softmax", 3);
     ]);

     the inputShape() method allows you to get the shape of your input.
     */
    inputShape(shapeConfig) {
        try {
            if (shapeConfig.features) {
                const features = shapeConfig.features;
                this.input_shape = null;
                return {
                    layer_name: "input_layer",
                    layer_size: features,
                    input_shape: null
                };
            } else if (shapeConfig.height && shapeConfig.width && shapeConfig.depth) {
                const { height, width, depth } = shapeConfig;

                return {
                    layer_name: "input_layer",
                    layer_size: height * width * depth,
                    input_shape: [height, width, depth]
                };
            } else {
                throw new Error(`[ERROR]------- Invalid input shape config`);
            }
        } catch (error) {
            console.error(error.message);
        }
    }

    /**
    * 
    *
    @method flatten()  
    Flattens the output of the last convolutional layer into a 1D array. This layer is crucial before connecting to fully connected layers as
    it bridges the gap between feature extraction part of your network and the connected layers.
    */
    flatten() {
        return {
            "layer_name":"flatten_layer",
            feedforward: (onGPU, input, weights=null, bias = null, current_layer) => {
                
                const flattened = input.flat(Infinity);

                //console.log('forward Flattened:',flattened)

                return {
                    outputs: flattened,
                    z_values: flattened,
                    incrementor_value: 0
                }
            },
            backpropagate: (onGPU, next_weights, next_delta, zs, layer_index, currentLayer) => {
                // Get the feature maps output from the previous convolutional layer

                // console.log('Weights',next_weights)
                const feature_maps = zs[layer_index - 1];
                const filters = feature_maps.length;
                const [H, W, D] = [
                    feature_maps[0].length,
                    feature_maps[0][0].length,
                    feature_maps[0][0][0].length
                ];

                // STEP 1: Compute the delta for the flatten layer (pre-flattened form)
                // multiply next_weights * next_delta
                // let flatten_delta = new Array(next_weights.length).fill(0);
                // for (let i = 0; i < next_weights.length; i++) {
                //     let sum = 0;
                //     for (let j = 0; j < next_weights[i].length; j++) {
                //         sum += next_weights[i][j] * next_delta[j];
                //     }
                //     flatten_delta[i] = sum;
                // }

                const flatten_delta = DeltaMatMul(next_weights, next_delta);
                // console.log('What the flatten layer receive during backprop',flatten_delta);

                // STEP 2: Reshape flatten_delta back to convolutional feature maps
                const reshapeFeatureMaps = [];
                let idx = 0;
                for (let f = 0; f < filters; f++) {
                    const feature_map = [];
                    for (let h = 0; h < H; h++) {
                        let row = [];
                        for (let w = 0; w < W; w++) {
                            let depthArr = [];
                            for (let d = 0; d < D; d++) {
                                depthArr.push(flatten_delta[idx++]);
                            }
                            row.push(depthArr);
                        }
                        feature_map.push(row);
                    }
                    reshapeFeatureMaps.push(feature_map);
                }

                const delta = StackFeatureMaps(reshapeFeatureMaps);

                return {
                    current_delta: delta,
                    decrementor_value: 0
                };
            }
            // backpropagate: (onGPU, next_weights, next_delta, zs, layer_index, currentLayer) => {
            //     // next_delta is already δ_flat

            //     // console.log('what the flatten received',next_delta)
            //     const flatten_delta = next_delta;

            //     const [F, H, W, D] = currentLayer.input_shape;

            //     const reshapeFeatureMaps = [];
            //     let idx = 0;

            //     for (let f = 0; f < F; f++) {
            //         const feature_map = [];
            //         for (let h = 0; h < H; h++) {
            //             const row = [];
            //             for (let w = 0; w < W; w++) {
            //                 const depthArr = [];
            //                 for (let d = 0; d < D; d++) {
            //                     depthArr.push(flatten_delta[idx++]);
            //                 }
            //                 row.push(depthArr);
            //             }
            //             feature_map.push(row);
            //         }
            //         reshapeFeatureMaps.push(feature_map);
            //     }

            //     return {
            //         current_delta: reshapeFeatureMaps,
            //         decrementor_value: 0
            //     };
            // }


        }
    }


    /**
     * Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will
     * build connected layers or multilayer perceptron
     * @param {String} activation specify the activation function for this layer (Available: sigmoid, relu, tanh, linear)
     * @param {Number} layer_size specify the number of neuron for this layer.
     * @throws {Error} When activation function is undefined (no activation is provided) or layer size is not provided or it's 0
     */
    connectedLayer(activation_function, layer_size) {
        try {

            if (!activation_function || !layer_size || layer_size <= 0) {
                throw new Error(`[ERROR]------- Layer Error | Activation function: ${activation_function} | layer size: ${layer_size}`);
            }

            let function_name = activation_function.toLowerCase();

            if (!activation[function_name] || !activation.derivatives[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                "layer_name":"connected_layer", 
                "activation_function":activation[function_name], 
                "derivative_activation_function":activation.derivatives[function_name],
                "layer_size":layer_size,
                feedforward: (onGPU, current_input, weights, biases, current_layer) => {
                
                    let input = current_input;
                    if (Array.isArray(input[0]) || Array.isArray(input[0][0])) {
                        input = input.flat(Infinity);                  
                    }
                    
                    
                    const z_values = MatMul(input, weights, biases);
                    const activation_function = activation[function_name];

                    let outputs;
                    outputs = activation_function(z_values);
                    
                    return {
                        outputs, 
                        z_values,
                        incrementor_value: 1
                    };
                },
                backpropagate: (onGPU, next_weights, next_delta, zs, layer_index, currentLayer) => {
                    const current_delta = DeltaMatMul(next_weights, next_delta).map((value, i) =>
                        value * activation.derivatives[function_name]([zs[layer_index][i]])
                    );

                    

                    return {
                        current_delta,
                        decrementor_value: 1
                    };
                }
            };
        }
        catch (error) {
            console.log(error.message);
        }
    }

    /**
     * 
     * Allows you to add convolutional layers in your model architecture in sequential building.
     * @method convolutional2D
     * @param {Number} filters - the number of filters for this convolutional layer. Produces the same number of output features
     * @param {Number} strides - It determines how much the filter overlaps with the input as it slides across.
     * @param {Array<Number>} kernel_size - the size of the kernel (or filter) that will slide and extracts input features
     * @param {String} activation_function - the activation function to be use for this layer
     * @param {String} padding - adds extra values (typically 0s) around the border of an input before applying a convolutional filter
     * @throws {Error} - if any of the parameters are invalid.
     *
     */
    convolutional2D(filters, strides, kernel_size, activation_function, padding) {
        try {
            if (!filters || filters <= 0) throw new Error(`[ERROR]-------- Filters cannot be empty, less than or equal to 0. Filters: ${filters}`);
            if (!strides || strides <= 0) throw new Error(`[ERROR]-------- Strides cannot be empty, less that or equal to 0. Strides: ${strides}`);
            if (!kernel_size || kernel_size.length == 0 || (kernel_size[0] <= 0 || kernel_size[1] <= 0)) throw new Error(`[ERROR]------- Kernels cannot be empty, nor it's height or width is less than or equal to 0. Kernel size: ${kernel_size}`);
            if (!activation_function || activation_function == undefined || activation_function == null || activation_function === "") throw new Error(`[ERROR]-------- activation_function cannot be empty, null or undefined.`);
            if (!padding || padding == undefined || padding == null || padding === "") throw new Error(`[ERROR]-------- Padding cannot be empty, null or undefined.`);

            // check if the padding is valid
            let paddings = ["same", "valid"];
            if (!paddings.includes(padding.toLowerCase())) {
                throw new Error(`[ERROR]------- ${padding.toLowerCase()} is invalid. Use 'same' or 'valid' only`);
            }

            // check if the activation function is valid
            const function_name = activation_function.toLowerCase();

            if (!activation[function_name] || !activation.derivatives[function_name]) {
                throw new Error(`[ERROR]------- Activation function '${function_name}' or its derivative not found or invalid,`);
            }

            return {
                "layer_name":"convolutional2D",
                "activation_function":activation[function_name],
                "derivative_activation_function":activation.derivatives[function_name],
                "kernel_size":kernel_size,
                "filters":filters,
                "padding":padding.toLowerCase(),
                "strides":strides,
                // kernels are also considered weights
                feedforward: (onGPU, input, weights, biases, current_layer) => {
                    
                    // 1. compute expected output tensor shape
                    const { OutputHeight, OutputWidth } = calculateTensorShape(input.length, input[0].length, weights[0].length, weights[0][0].length, weights[0][0][0].length, current_layer.strides, current_layer.padding);

                    // 2. Get padding sizes based on ORIGINAL input
                    const { top, bottom, left, right } = getPaddingSizes(input.length, input[0].length, weights[0].length, weights[0][0].length, current_layer.strides, current_layer.padding);
                    
                    // 3. Apply the padding
                    const paddedInput = applyPadding(input, top, bottom, left, right);

                    // 4. Perform the convolve operation using the shapes calculated in step 1
                    const new_feature_maps = Convolve(strides, paddedInput, weights, biases, OutputHeight, OutputWidth);


                    const stacked = StackFeatureMaps(new_feature_maps);
                                 
                    const z_values = new_feature_maps; // z_values are the new feature maps before activation

                    //console.log(z_values[0].length);

                    // activate each depth input using the given activation function
                    const activation_function = activation[function_name];
                    const outputs = [];
                    stacked.forEach((row, i) => {
                        const innerRow = row.map(cell => activation_function(cell));
                        outputs.push(innerRow);
                    });

                    return {
                        outputs,
                        z_values,
                        incrementor_value: 1
                    };
                },

                backpropagate: (onGPU, next_weights, next_delta, zs, layer_index, currentLayer, weights, activations) => {
                
                    let input = next_delta;
                    let kernels = weights;
                    
                        
                        // If this is the input to the first convolutional layer, stop here
                    if (layer_index == 0) {
                        const current_Z = StackFeatureMaps(zs[layer_index]);
                        const inputH = current_Z.length;
                        const inputW = current_Z[0].length;

                        kernels = weights[layer_index];
                        input = toTensor(input, [activations[layer_index].length, activations[layer_index][0].length, activations[layer_index][0][0].length]);
                        const KH = kernels[0].length;
                        const KW = kernels[0][0].length;
                        // dilate the input
                        const dilated_input = DilateInput(input, strides);

                        // apply padding
                        const padH = KH - 1 + 1;
                        const padW = KW - 1 + 1;
                        const padded_dilated_input = applyPadding(dilated_input, padH, padH, padW, padW);

                        const deltaConv = ConvolveDelta(padded_dilated_input, kernels, inputH, inputW);

                        //console.log(`Output delta shape of this convolutional layer ${layer_index+1},`, deltaConv.length, deltaConv[0].length, deltaConv[0][0].length, '\n');
                    
                        return {
                            current_delta: deltaConv,
                            decrementor_value: 1
                        };
                    }

                    const current_Z = StackFeatureMaps(zs[layer_index-1]);

                    if (!Array.isArray(input[0][0])) {

                        const [H, W, D] = [current_Z.length, current_Z[0].length, current_Z[0][0].length];

                        const flatten_delta = DeltaMatMul(next_weights, input);
                        
                        // reshape the flattened_Delta
                        input = toTensor(flatten_delta, [H, W, D]);

                    }

                    const inputH = current_Z.length;
                    const inputW = current_Z[0].length;

                    kernels = weights[layer_index];

                    const KH = kernels[0].length;
                    const KW = kernels[0][0].length;
                    // dilate the input

                    // if (input.flat(Infinity).some(isNaN)) {
                    //     console.log(`Input already has NaNs at layer ${layer_index+1}`);
                    // }

                    const dilated_input = DilateInput(input, strides);

                    // apply padding
                    const padH = KH - 1 + 1;
                    const padW = KW - 1 + 1;
                    const padded_dilated_input = applyPadding(dilated_input, padH, padH, padW, padW);

                    // console.log(`Layer ${layer_index+1}`);
                    // console.log('kernel shape', kernels.length, kernels[0].length, kernels[0][0].length, kernels[0][0][0].length);
                    // console.log('Padded dilated delta shape', padded_dilated_input.length, padded_dilated_input[0].length, padded_dilated_input[0][0].length);
                    // console.log(`Input Height: ${inputH}, Input width: ${inputW}`);
                
                    const deltaConv = ConvolveDelta(padded_dilated_input, kernels, inputH, inputW, layer_index+1);
                    // if (dilated_input.flat(Infinity).some(isNaN)) console.log(`dilated_input conv already has NaNs in layer ${layer_index+1}`);
                    // if (padded_dilated_input.flat(Infinity).some(isNaN)) console.log(`padded_dilated_input already has NaNs in layer ${layer_index+1}`);
                    // if (deltaConv.flat(Infinity).some(isNaN)) console.log(`output delta conv already has NaNs in layer ${layer_index+1}\n`);

                    // const flattened = padded_dilated_input.flat(Infinity);

                    // if (flattened.some(isNaN)) {
                    //     console.log("There are NaN's in padded_dilated_input delta input after reshaping and convolving at",layer_index+1);
                    //     throw new Error();
                    // }

                    if (deltaConv.flat(Infinity).some(isNaN)) throw new Error(`Layer ${layer_index+1} has NaNs after delta convolution`)

                    const z = current_Z;

                    const dActivation = activation.derivatives[function_name];

                    // apply the derivative activation function for all zs
                    const dAct_Z = z.map(row => row.map(cell => dActivation(cell)));

                    // multiply input x dAct_Z
                    const outputDelta = deltaConv.map((row, h) => row.map((cell, w) => cell.map((val, c) => val * dAct_Z[h][w][c])));

                    //console.log(`Output delta shape of this convolutional layer ${layer_index+1},`, outputDelta.length, outputDelta[0].length, outputDelta[0][0].length, '\n');

                    return {
                        current_delta: outputDelta,
                        decrementor_value: 1
                    };
                }


            }

        }
        catch (error) {
            console.error(error);
        }
    }
}

module.exports = Layers;
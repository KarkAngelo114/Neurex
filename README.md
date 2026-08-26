
![Alt text](https://res.cloudinary.com/ddgfmkjjm/image/upload/v1751615537/NodeJS-neurex_dky5vh.png)

[![NPM](https://nodei.co/npm/neurex.svg)](https://nodei.co/npm/neurex/)
[![NPM](https://nodei.co/npm/neurex.svg?style=shields&data=n,v,u,d)](https://nodei.co/npm/neurex/)


## How to get started

1. Ensure you have NodeJS installed on your machine. If you haven't installed it yet, download it here: https://nodejs.org/en/download
2. Create a new folder and navigate to your created folder in your terminal.
3. Once you're inside your project directory, install Neurex using this command:

```bash

npm install neurex

```


## Documentation
Checkout the documentation for full API reference, live demos, and some starter examples [here](https://neurex-documentation.vercel.app/).

# Neurex
Neurex is a Javascript-based, GPU-Accelerated, deep learning library for Node.js. It supports training on CPU and can also utilized GPU with the help of [OpenCL](https://github.com/KhronosGroup/OpenCL-Headers) if available. This library supports:

1. 🧠 Easy model building through sequential stacking ✅
2. 🛠️ Both CommonJS and ES module importing ✅
3. 🔃 Retraining and transfer learning ✅
4. ⚡ GPU acceleration for faster training ✅

## Why use Neurex
1. Easy implementation - intuitive API calls. No need to fight with the API design
2. Abstracted complexities - Intuitive API that handles the heavy lifting of backpropagation and weight initialization, allowing you to focus on architecture.
3. Educational - Good for experimenting or learning how to build Neural networks
4. Use vs See - Others just let you use their predefined networks. Neurex lets you build and see the network to train, allowing you to design your model for your own use case.


## Build your model sequentially
Use built-in layers from the `Layers` class to build your model.

```Javascript
const { Neurex, Layers } = require('neurex');

(async () => {
    const nrx = new Neurex();
    const layer = new Layers();

    // stack layers in sequential order
    nrx.sequentialBuild([
        layer.inputShape({ features: 3 }),
        layer.connectedLayer(5), // layer size: 5, activation: relu (by default)
        layer.connectedLayer(5), // layer size: 5, activation: relu (by default)
        layer.connectedLayer(5), // layer size: 5, activation: relu (by default)
        layer.connectedLayer(10, 'softmax')
    ]);
})();
```


### Built-in layers:
The `Layers` class acts as a factory for generating neural network layer configurations. Here are some layers that are avaulable to use:

connectedLayer(`layer_size: number, activation: string, useBias: boolean`)
- Allows you to build a layer with number of neurons and the activation function to use in a layer. Stacking more layers will build connected layers or multilayer perceptron

```JavaScript
layer.connectedLayer(5, 'tanh');
```

convolutionalLayer(`filters: number, strides: number, kernel_size: number[], activation_function: string, padding: string, useBias: boolean`)
-  Allows you to add convolutional layers in your model architecture in sequential building.
```JavaScript
layer.convolutionalLayer(12, 1, [3, 3], 'relu', 'same'); // or use 'valid'
```

embeddingLayer(`vocabSize: number, embeddingDim: number, maxSequenceLength: number`)
- Creates an embedding layer for token encoding.
```JavaScript
layer.embeddingLayer(5000, 50, 10)
```

maxPooling(`poolSize: number[], strides: number, padding: string`)
- is use for downsampling operation that reduces the spatial dimensions of an input tensor by taking the maximum value over a defined sliding window

```JavaScript
layer.maxPooling([2, 2], 2, 'same'); // or use 'valid'
```

recurrentCell(`units: Number, activation_function: string, return_sequence: boolean, useBias: boolean`)
- is the fundamental building block of a Recurrent Neural Network (RNN) designed to process sequential data. It maintains an internal `memory` by taking its output from the previous time step and feeding it back into itself alongside the new input.

```JavaScript
layer.recurrentCell(18, 'tanh', true); // or false if the next layer in the stack is not recurrent
```

reshape(`targetSize: number[]`);
- changes the dimensions (shape) of the data passing through it without changing the data values. This acts as the `connector` to bridge data from different layers (e.g: from connected layer to convolutional layer). 
```JavaScript
layer.reshape([28, 28, 1]);
```

transConvLayer(`filters: number, strides: number, kernel_size: number[], activation_function: string,  padding: string, inputShape: number[], useBias: boolean`)
- `transConv` (or transpose convolution) is a specialized convolutional layer that upsamples incoming tensor map, which does the opposite of the normal convolution
```JavaScript
layer.transConvLayer(3, 2, [3, 3], 'linear', 'same', [56, 56, 5], false),
```

simpleAttention(`useBias: boolean`)
- `simpleAttention` is the implementation of an attention layer in its simpliest and basic form.
```JavaScript
layer.simpleAttention()
```

For more info about layers, check the official [documentation](https://neurex-documentation.vercel.app/javascript-nodejs#layers).


## Sample usage - training a XOR 
Here's an example on how you can use `Neurex` to train on XOR problem.

```Javascript
const {Neurex, Layers, Adam, SGD, stepDecay} = require('neurex');

const nrx = new Neurex();
const layer = new Layers();


(async () => {
    const trainX = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    const trainY = [
        [0],
        [1],
        [1],
        [0]
    ];

    // configurations
    nrx.configure({
        optimizer: Adam(), // use built-in optimizers or plug your own optimizer function here!
        lr_scheduler: stepDecay() // use built-in scheduler or plug your own!
        learning_rate:0.0001, // learning rate value
        checkpoint_per_epoch: 10, // if set, it saved the model every N epoch.
        mode: "cpu", // "gpu" or "auto"
        clip_norm_value: 5.0, // value to clip gradients
        // onFloat32Module: true, if set to true, it won't use its native bindings and rather use the JS implementation fallback

        // config to use Neurex auto-switch optimizer feature
        onChange_optimizer: {
            optimizer: SGD(), // optimizer to use (use the built-in or plug your own)
            targetEpoch: 50 // when it reach the specific target epoch, it will swap the optimizer with the one you set in this config
        }
    });

    // stack layers in sequential order
    nrx.sequentialBuild([
        layer.inputShape({ features:2 }),
        layer.connectedLayer(4), // layer size: 4, activation: relu (by default)
        layer.connectedLayer(1, 'sigmoid')
    ]);


    // you can show the summary of your model by calling modelSummary()
    nrx.modelSummary();

    // train the model
    await nrx.train(trainX, trainY, 'binary_cross_entropy', 1000, 2);

    // save model
    nrx.saveModel('model'); // this will be saved as model.nrx

    // predict
    const predictions = await nrx.predict(trainX);
    console.log(pedictions); // predicted outputs are in float32array. You may convert it to normal JS array if you need
    /*
    * Example:
    * [
    *   Float32Array(1) [ 0.2107422947883606 ]
    * ]
    */
})();
```




## Saving, loading, popping and adding new layers for transfer learning

Saving models is very straightforward. 

```JavaScript
    (async () => {
        
        const {Neurex, Layers} = require('neurex');

        const nrx = new Neurex();

        // ... data preprocessing, layer stacking, configurations, train()

        await nrx.saveModel()
    })()
```

Your model will be saved in a binary file which can be use later on.

_Note: `nrx` (or neurex) models are exclusive model file format in `Neurex` only._

To load `.nrx` models, you can use `loadSavedModel()`. This will load and recontruct your trained model

```Javascript
const { Neurex, Layers } = require('neurex');

(async () => {
    const nrx = new Neurex();
    const layer = new Layers();

    await nrx.loadSavedModel("model.nrx");

    nrx.modelSummary(); // prints the model summary
})();

```

To remove and add a new layer, use `pop()` and `add_layer()`. These methods are essentials if you do `transfer learning`

```Javascript
const { Neurex, Layers } = require('neurex');

(async () => {
    const nrx = new Neurex();
    const layer = new Layers();

    await nrx.loadSavedModel("model.nrx");
    nrx.pop(); 

    // calling more pop() will remove every last layer
    // nrx.pop(); 
    // nrx.pop(); 
    // nrx.pop(); 
    nrx.add_layer(layer.connectedLayer(3,'softmax')) // append a new layer with untrained parameters

    nrx.modelSummary(); // prints the model summary
})();
```

## Use built-in templates

Want to train a model immediately? The `templates` module offers curated templates you can use which you can drop in to the `sequentialBuild()` method.

```Javascript

const { Neurex, Layers, templates } = require('neurex');

(() => {
    const nrx = new Neurex();
    const layer = new Layers();

    nrx.sequentialBuild([
        layer.inputShape({features: 2}),
        // drop in a connected network having 3 hidden layers, 5 neurons each
        ...templates.simpleNeuralNetwork(),
        layer.connectedLayer(1, 'sigmoid')
    ])
})();
``` 

```Javascript
const { Neurex, Layers, templates } = require('neurex');

(() => {
    const nrx = new Neurex();
    const layer = new Layers();

    nrx.sequentialBuild([
        layer.inputShape({features: 2}),
        // drop in a convolutional network. If "isHeadless" parameter is set to true, the funnel-shape connected layer will be removed. Default is `false`
        ...templates.simpleCNN(isHeadless = true),
        layer.recurrentCell(18, 'tanh', true),
        layer.recurrentCell(18, 'tanh', true),
        layer.recurrentCell(18, 'tanh'),
        layer.connectedLayer(1, 'sigmoid')
    ])
})();
``` 


```Javascript
const { Neurex, Layers, templates } = require('neurex');

(() => {
    const nrx = new Neurex();
    const layer = new Layers();

    nrx.sequentialBuild([
        layer.embeddingLayer(5000, 50, 10),
        ...templates.vanillaRNN(18, 'reu'), // uses 3 recurrent cells, each has 3 units be default and tanh activation. All uses `return_sequences = true`
        layer.connectedLayer(1, 'sigmoid')
    ])
})();
```

Learn more about neural network templates [here](https://neurex-documentation.vercel.app/javascript-nodejs#templates).

# Experiment and plug your own optimizer and learning rate scheduler (in Development)
Thanks to the updated core engine and flexible API, you can now write and plug your own optimizer and learning rate scheduler!

```Javascript
const { Neurex } = require('neurex');

function MyOptimizer() {
    return function AwesomeOptimizer(data) {
        // destructure to extract the data use for computation
        // tip: log the whole data object to know what the engine gives you for optimizing params
        const { params, grads, state: state = {}, lr } = data; 
        

        // your actual implementation...


        // ALWAYS RETURN UPDATED PARAM AND STATE
        return {
            params: params,
            state: state,
        };
    };
}

function CustomScheduler() {
    return function scheduler(data) {
        // destructure to extract the data use for computation
        // tip: log the whole data object to know what the engine gives you for calculating new learning rate
        const {current_epoch, learning_rate, previousEpochLoss } = data;

        // your actual implementation...

        // ALWAYS RETURN NEWLY CALCULATED LEARNING RATE
        return updated_learning_rate;
    }
}


(() => {
    const nrx = new Neurex();

    nrx.configure({
        optimizer: MyOptimizer(),
        lr_scheduler: CustomScheduler(),
        /* other configs */
    });
})();
```

# Use pluggable monitoring tools (in Development)
Use monitoring tool plugins to monitor training in real-time!

```JavaScript
const { Neurex, Layers, lossVisualizer, modelVisualizer, lossLandscapeVisualizer } = require('neurex');

(async () => {
    const nrx = new Neurex();
    const layer = new Layers();
    
    nrx.sequentialBuild([
        layer.inputShape({height: 28, width: 28, depth: 1}),
        ...templates.simpleCNN(), // conv [3, 3] stride = 1 "same" -> maxPool [2, 2] stride = 2 "valid" -> conv [3, 3] stride = 1 "same" -> maxPool [2, 2] stride = 2 "valid" -> dense: 128 -> 64 -> 32
        layer.connectedLayer(10, 'softmax')
    ]);

    nrx.configure({
        /* ... other configs */
        visualizerPlugins: [
            modelVisualizer(),
            lossVisualizer(),
            lossLandscapeVisualizer()
        ]
    })

    await nrx.train(X, Y, 'categorical_cross_entropy', 1000, 12);

})()
```

![Dashboard](https://res.cloudinary.com/ddgfmkjjm/image/upload/v1787725595/Screenshot_2026-08-06_115813_ksqvw9.png)
![Dashboard](https://res.cloudinary.com/ddgfmkjjm/image/upload/v1785989320/Screenshot_2026-08-06_115739_fcrzv4.png)
![Dashboard](https://res.cloudinary.com/ddgfmkjjm/image/upload/v1785989320/Screenshot_2026-08-06_115828_tjt5zi.png)

# Test the Experimental Upcoming Updates 🔥
If you'd like to try the upcoming major updates before it is officially released on NPM, you can install the latest development version directly from GitHub.

## Install from GitHub

```bash
npm install git+https://github.com/KarkAngelo114/Neurex.git
```

## Notes

* APIs may change without notice
* Some features may be incomplete
* Documentation may lag behind implementation
* Expect frequent updates and fixes

This is mainly intended for:

* early adopters
* contributors
* testers
* developers who want access to the newest features

Feedback and bug reports are highly appreciated 🙌

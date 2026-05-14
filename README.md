
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

1. 🧠 Mix and Match; train CNN + ANN or just ANN ✅
2. 🛠️ Both CommonJS and ES module importing ✅
3. 🔃 Retraining and transfer learning ✅
4. ⚡ GPU acceleration for faster training ✅

## Why use Neurex
1. Easy implementation - intuitive API calls. No need to fight with the API design
2. Abstracted complexities - Intuitive API that handles the heavy lifting of backpropagation and weight initialization, allowing you to focus on architecture.
3. Educational - Good for experimenting or learning how to build Neural networks
4. Use vs See - Others just let you use their predefined networks. Neurex lets you build and see the network to train, allowing you to design your model for your own use case.

## Sample usage - training a XOR 
Here's an example on how you can use `Neurex` to train on XOR problem.

```Javascript
const {Neurex, Layers} = require('neurex');

const nrx = new Neurex();
const layer = new Layers();

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
    optimizer:'adam',
    learning_rate:0.1,
    checkpoint_per_epoch: 100, // if you want to save the model for every N epochs (let's say every 100 epochs like in this example)
    mode:"cpu", /* "gpu" or "auto" */
});

// stack layers in sequential order
nrx.sequentialBuild([
    layer.inputShape({ features:2 }),
    layer.connectedLayer('relu', 4),
    layer.connectedLayer('sigmoid',1)
]);


// you can show the summary of your model by calling modelSummary()
nrx.modelSummary();

// train the model
nrx.train(trainX, trainY, 'binary_cross_entropy', 1000, 2);

// predict
const predictions = nrx.predict(trainX);
trainX.forEach((input, i) => {
    const pred_val = predictions[i][0] > 0.5 ? 1 : 0; // since it's binary classiification. Predicted outputs are between 0 and 1
    console.log(`Input: ${input} Predicted value: ${pred_val} | Raw output: ${predictions[i][0]} | Actual: ${trainY[i][0]}`);
});

```

You can also used predefined neural network templates which you can drop in to the `sequentialBuild()`

```Javascript

const { Neurex, Layers, templates } = require('neurex');

(() => {
    const nrx = new Neurex();
    const layer = new Layers();

    nrx.sequentialBuild([
        layer.inputShape({features: 2}),
        // drop in a connected network having 3 hidden layers, 5 neurons each
        ...templates.simpleNeuralNetwork(),
        layer.connectedLayer('sigmoid',1)
    ])
})();
```

Learn more about neural network templates [here](https://neurex-documentation.vercel.app/).

# Test the Experimental Upcoming Updates 🔥
> ⚠️ This version is currently under active development and may contain breaking changes, bugs, or incomplete features.

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



> [!NOTE]
> Everything described above applies to the upcoming latest version of `Neurex` and may differ from the current stable NPM release.


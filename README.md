
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
4. After successful installation, you can now train ANN models and do predictions in your backend applications.

## Documentation
Checkout the documentation how to get started and some starter examples [here](https://neurex-documentation.vercel.app/).

# Neurex
Neurex is a Javascript-based neural network library for Node.js, designed to be fully trainable and easy to integrate into your applications.

## Sample usage - training a XOR 
Here's an example on you can use `Neurex` to train on XOR problem.

```Javascript
const {Neurex, Layers} = require('neurex');

const nrx = new Neurex();
const layer = new layers();

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
    learning_rate:0.1
});

// stack layers in sequential order
nrx.sequentialBuild([
    layer.inputShape({ features:2 }),
    layer.connectedLayer('relu', 4),
    layer.connectedLayer('sigmoid',1)
]);

// build the model
nrx.build();

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
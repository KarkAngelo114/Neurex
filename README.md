
![Alt text](https://res.cloudinary.com/ddgfmkjjm/image/upload/v1751615537/NodeJS-neurex_dky5vh.png)

## Neurex
Neurex is a JavaScript-based neural network library for Node.js, designed to be fully trainable and easy to integrate into your applications.

## How to get started

1. Ensure you have NodeJS installed on your machine. If you haven't installed it yet, download it here: https://nodejs.org/en/download
2. Create a new folder and navigate to your created folder in your terminal.
3. Once you're inside your project directory, install Neurex using this command:

```bash

npm install neurex

```
4. After successful installation, you can now train ANN models and do predictions in your backend applications.

## Documentation
Checkout the documentation how to get started [here](https://neurex-documentation.vercel.app/).

## Release Notes
### GPU Acceleration Powered by gpu.js
This release introduces a substantial performance boost by enabling GPU acceleration for core neural network operations. [gpu.js](https://github.com/gpujs/gpu.js) is integrated to transparently transpile computationally intensive JavaScript functions (such as matrix multiplications, backpropagation delta calculations, activation functions, and optimizer updates) into optimized code that runs directly on your Graphics Processing Unit (GPU). This significantly reduces training times and enhances overall efficiency.

> [!Note]
> - can only do ANN modelling.
> - Falls back to pure Javascript operations if GPU is not available.
> - exported models (.nrx) are exclusive only in Neurex. Trained models cannot be loaded to other libraries or frameworks.
> - Starting on July 29, 2025, older versions will be deprecated.

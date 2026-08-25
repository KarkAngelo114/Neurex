# v0.1.0 (In Development)
### What's New
- Added `recurrentCell` to support RNN modeling. See [index.d.ts](https://github.com/KarkAngelo114/Neurex/blob/main/index.d.ts) for more info.
- Added `transConv` layer. See [index.d.ts](https://github.com/KarkAngelo114/Neurex/blob/main/index.d.ts) for more info.
- Addded `reshape` layer. See [index.d.ts](https://github.com/KarkAngelo114/Neurex/blob/main/index.d.ts) for more info.
- Added `simpleAttention` layer. See [index.d.ts](https://github.com/KarkAngelo114/Neurex/blob/main/index.d.ts) for more info.
- added learning rate schedulers
- added auto-switching feature via `configure()` method for optimizer switching amidst model training. See [here](https://neurex-documentation.vercel.app/javascript-nodejs#configure)
- You can now add you own optimizers and learning rate scheduler. See the latest documentation about [optimizers](https://neurex-documentation.vercel.app/javascript-nodejs#optimizers), and [lr_schedulers](https://neurex-documentation.vercel.app/javascript-nodejs#schedulers), and how to plug in your own.
- added plugins for visualizer tools. Built-in visualizers are `lossVisualizer()`, `lossLandscapeVisualizer()`, and `modelVisualizer()`

### Fixes
- fixed all derivative activation functions.

### What's Updated
- arguments on `connectedLayer()` has been flipped. Instead of `connectedLayer(activation_func, layer_size)`, it's now `connectedLayer(layer_size, activation_func)`. See the updated the documentation [here](https://neurex-documentation.vercel.app/javascript-nodejs#layers).
- when using `load_images_from_directory()`, you can pass a string value in the `label_mode` argument. The label mode to use depends on the loss function you will going to use for training. See the updated documentation [here](https://neurex-documentation.vercel.app/javascript-nodejs#load_images_from_directory).
- in the `optimizer` property when setting config, it can only now accepts factory functions rather than string name of an optimizer allowing you to plug your own custom optimizer.

### Breaking Changes
- because the `optimizer` property when setting config can only now accepts factory functions rather than string name of an optimizer, any training scripts that uses the same config will break.
- supported model version has been bumped to `NRX4`. Old models cannot be loaded. 
- due to arrangement of arguments on `connectedLayer()`, ensure that your training script/s also follows the format.


# v0.0.9 (Latest)
### What's New
- major overhaul of the entire core functionalities of the library to use float32array
- 2x performance boost due to type arrays
- uses native bindings written in C++ and are already precompiled so that you don't have to compile again (source code is in different repository)
- introduced CNN layers. Now supports training Convolutional Neural Networks
- added pooling layers
- allows retraining and transfer learning
- introduced Embedding layer

### Breaking Changes
- due to fully transition to float32array and the major overhaul of the codebase, models that are trained using later version of `Neurex` will no longer be supported
- `Interpreter` class has been removed. You can now directly use the loading function with the main class.


# v0.0.7 
### What's New
- introduced sequential stacking (via `sequentialBuild`)
- more internal functions are modular

# v0.0.6
### What's New
- now supports multi-regression, binary and multi-class classification
- proper serializaion of saving models (in .nrx format)

### Breaking Changes
- models trained on the later version cannot be loaded (loading models from JSON is no longer supported)

# v0.0.4 (deprecated)
### What's New
- update fixes on the `CsvDataHandler` module


# v0.0.3 (deprecated)
### What's New
- introduced `CsvDataHandler` module. Now can work with CSV tabular datasets
- still limited for use

# v0.0.1 (deprecated)
### What's New
- first publish
- limited use
- can only train on regression task
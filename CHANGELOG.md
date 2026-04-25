# v0.0.9 (in development)
### What's New
- major overhaul of the entire core functionalities of the library to use float32array
- 2x performance boost due to type arrays
- uses native bindings written in C++ and are already precompiled so that you don't have to compile again (source code is in different repository)
- introduced CNN layers. Now supports training Convolutional Neural Networks in the near future
- added pooling layers
- allows retraining and transfer learning

### Breaking Changes
- due to fully transition to float32array and the major overhaul of the codebase, models that are trained using later version of `Neurex` will no longer be supported
- `Interpreter` class has been removed. You can now directly use the loading function with the main class.


# v0.0.7 (latest)
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
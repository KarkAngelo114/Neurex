
![Alt text](https://res.cloudinary.com/ddgfmkjjm/image/upload/v1751615537/NodeJS-neurex_dky5vh.png)

## Neurex
Neurex is a JavaScript-based neural network library for Node.js, designed to be fully trainable and easy to integrate into your applications. It's implementation is inspired from other ML-libraries.

## How to get started

1. Ensure you have NodeJS installed on your machine. If you haven't installed it yet, download it here: https://nodejs.org/en/download
2. Create a new folder and navigate to your created folder in your terminal.
3. Once you're inside your project directory, install Neurex using this command:

```bash

npm install neurex

```
4. After successful installation, you can now train ANN models and do predictions in your backend applications.

## Demo

Below is the demo how to use Neurex

## Importing Neurex's modules

Within Neurex, there are modules to import. For this demo, we'll be importing Neurex's core where the training logic lives, split module to split dataset into training and testing, regression metrics module to evaluate our trained model, and the built-in dataset (HousePricing dataset). For this demo, will be predicting house price.

```Javascript

const Neurex = require('neurex/core');
const split_dataset = require('neurex/preprocessor/split');
const evaluate = require('neurex/metrics/regression_metrics');
const HousePricing = require('neurex/datasets/HousePricingDataset');

```

## Preprocessing

We have imported our built-in dataset. Now we can use it to train our first model. But we can't just shove the entire dataset, so we split it to 80% training and 20% testing. The 80% will be use for training and the remaining 20% will be use for evaluation later (it will serve as an unseen data)

```Javascript

const {X_train, Y_train, X_test, Y_test} = split_dataset(HousePricing.trainX, HousePricing.trainY, 0.2);

```

We have split our dataset into 80% training and 20% testing set (0.2 is the test size ratio, so the remaining 0.8 will go to train set). X_train are the train set and its corresponding target values is the Y_train. X_test and Y_test will be use for evaluation.

## Building layers

We have successfully imported our dataset, split it into train and test sets, now we build our network. To do this, you would need first to create an instance of Neurex.

```Javascript

const Neurex = require('neurex/core');
/* other imports */

/* preprocesses */

const model = new Neurex();

```

Now, we can use the instance variable (model). To get started in building our network, we need -- the input layer, hidden layer, and an output layer. For input layer, we use inputShape() method and pass the X_train to get the input size (or the number of input neurons in the input layer).

```Javascript

/* Imports and preprocesses */

const model = new Neurex();

model.inputShape(X_train);

```

Next, we build the hidden layes using construct_layer()

```Javascript

model.construct_layer("relu", 20);
model.construct_layer("relu", 5);
model.construct_layer("relu", 3);

```

And for the output layer, we still going to use the construct_layer()

```Javascript

model.construct_layer("linear", 1);

```

As you can see, we built a network with 4 layers -- three hidden layers having same activation function and different numbers of hidden neurons on each layer. The output layer has an activation function of "linear" and an output size of 1. You can inspect you model's architecture by using the modelSummary() before training.

```Javascript

model.modelSummary();

```

## Training

If you're ready to train your model, you can use the train() to start the training process. For this demo, before we train our model, we configure it first via configure()

```Javascript

model.configure({
    learning_rate: 0.001,
    optimizer: 'sgd'
});

```

Here, we use "sgd" (stochastic gradient descent) optimizer and a learning rate of 0.001. After configuring our network, we can now train our model using train(). We simply pass the X_train, Y_train, loss function to use, epoch, and batch size.

```Javascript

/* importing of modules, preprocesses, creating instances, building the network */

model.train(X_train, Y_train, "mse", 50000, 10);

```

Here, we train the model for 50k epochs and has a batch size of 10

And to save your model after training, you can use saveModel(). For this demo, I named my model "HousePrice". The model will be saved as a JSON file along with a metadata.json. The HousePricing.json is the actual model file, inspecting the contents of the model file is the stored parameters of your model. Note: the trained model can only be use by Neurex. You can't use the trained model to load it on other machine-learning libraries.

```Javascript

model.saveModel('HousePrice');

```

## Predicting

If you want to evaluate your model immediately after training, you can use Neurex's predict() and call evaluate() from regression_metrics.

```Javascript

/* previous processes */

const predictions = model.predict(X_test);

evaluate(predictions, Y_test);

```

As you can see, we use the X_test, and Y_test from before where these are the "unseen" data, the model haven't seen yet during training.

For the full demo script:

```Javascript
const Neurex = require('neurex/core');
const evaluate = require('neurex/metrics/regression_metrics');
const HousePricing = require('neurex/datasets/HousePricingDataset');
const split_dataset = require('neurex/preprocessor/split');

const model = new Neurex();

const {X_train, Y_train, X_test, Y_test} = split_dataset(HousePricing.trainX, HousePricing.trainY, 0.2);

model.inputShape(X_train);

model.construct_layer("relu", 20);
model.construct_layer("relu", 5);
model.construct_layer("relu", 3);
model.construct_layer('linear', 1);

model.configure({
    learning_rate: 0.001,
    optimizer: 'sgd'
    // randMin: -0.1, // minimum range for initializing weights and biases
    // randMax: 0.1 // maximum range for initializing weights and biases
});

model.train(X_train, Y_train, "mse", 50000, 10);

model.saveModel('HousePrice');

const predictions = model.predict(X_test);

evaluate(predictions, Y_test);

```

Once ready, run your script on your terminal/CMD to start training (mine is main.js), and watch the model learns.

```bash

node main

```

## Loading the trained model

Neurex has it's own interpreter module where you can use it for your application.

```Javascript

const Interpreter = require('neurex/interpreter');

```

Then we create an instance of the Interpreter class

```Javascript

const Interpreter = require('neurex/interpreter');

const interpreter = new Interpreter();

```

then using the instance variable (interpreter), you can use loadSavedModel() and predict().

Loading the trained model:

```Javascript

interpreter.loadSavedModel('HousePrice.json');

```

To use predict():

```Javascript

/* how you preprocess your data to be use in the predict() */

const predictions = interpreter.predict(your_input_data_here);

/* your application's logic how will you make the use of the predicted data */

```

When loading your model, ensure that is not modified and it is in the same location where your script that uses the interpreter is.

## Bonus

Experiment with your own dataset using Neurex!

## Update

NEW - the csv_loader module allows you to load your dataset in a CSV (comma separated values)


Here's how to use it:

1. Import the module:
```Javascript

const csv_loader = require('neurex/preprocessor/csv_loader');

```

2. Create a new variable instance:
```Javascript

const reader = new csv_loader();

```

3. Extract data using read_csv(). Passing a boolean value allows you to remove the rows of column names. This is because when you extract the data in a CSV file, all are converted into rows. Row1 (indexed as 0 in the array) are the column names. Row2 onwards are the data under those column names. Try logging the returned result without passing first a boolean value and then pass the boolean value to see the difference.

```Javascript

const dataset = reader.read_csv('awesome-neurex.csv', true);

```

4. The csv_loader module also offers different tools to use to manipulate your structure data like:

        - RowsToInt() -> You can convert data on every rows into numerical data using RowsToInt(). Ensure that there are no values that cannot be converted to numbers (non-numeric values).

        ```Javascript
        const data = reader.RowsToInt(dataset);

        ```

        - removeColumn() -> allows you to remove an entire column and it's data by specifying the column name. You would need to pass the    extracted data from your CSV. Note that when using this function, this will alter your extracted dataset.

        ```Javascript
        const modifiedDataset = reader.removeColumn("column_name", data);
        ```

        - getRowElements() ->  get all elements on all rows. Passing a numerical value sets a range to select elements within a row. This won't alter the original structure of your extracted dataset

        ```Javascript
        /*
            Example:
            [
                [1, 2, 3, 4, 5, 6, 7, ...],
                [1, 2, 3, 4, 5, 6, 7, ...],
                [1, 2, 3, 4, 5, 6, 7, ...],
                [1, 2, 3, 4, 5, 6, 7, ...],
                ...
            ]
        */
        
        const returnedRows = reader.getRowElements(5, modifiedDataset);

        /*Output:
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                ...
            ]
        
        */

        ```

        - extractColumn() -> select all data under the specified column. This alters your extracted dataset. Returns a 1D array

        ```Javascript
        const columnData = reader.extractColumn("column_name", modifiedDataset);
        ```

When using csv_loader and it's tools in manipulating extracted data, it is a good practice to always do logging to keep track what has been change and the updated structured of your extracted data.

## Notes

This libarary is:

- open source

- this trainable neural network library doesn't rely on any dependencies.

- doesn't have any other dependencies to install along with Neurex

- when saving models, it is in the form of JSON file. You can view the contents of your actual model, and though it is in human-readable form, it should not be modified. Attempting to do so will cause consequences (eg: incorrect predictions, shape-related erros, etc.).

- saved models can "only be use in Neurex" library. Saved models cannot be loaded using any other frameworks/libraries in "any" other programming languages (except the library itself and the language used to build the library) to do inference predictions or retraining your model.

>> Currently, this trainable neural network can only do regression tasks. But soon, I will extend it's capabilities.

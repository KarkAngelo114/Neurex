
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

A new module has been added allowing you to work with your dataset in CSV format. Using CsvDataHandler, you can manipulate your dataset with ease for data preprocessing and cleaning.

Here's an example how to do it:

Import CsvDataHandler module and create an instance:

```Javascript
const CsvDataHandler = require('neurex/preprocessor/CsvDataHandler');

const csv = new CsvDataHandler();
```

Load your CSV file on read_csv() method:

```Javascript
const dataset = csv.read_csv('my-awesome-dataset.csv');
```

You can view your loaded dataset in a tabularized format using tabularize() and simply passing a data to view. This is very useful especially if you want to keep track what is the current structure of your dataset along the process.

```Javascript
const dataset = csv.read_csv('my-awesome-dataset.csv');
csv.tabularize(dataset);
```

If your working with numerical data, consider using rowsToInt() method. This is because when you loaded your CSV dataset, all cell elements on all rows are strings. You can view the changes by logging the first result of the process and after converting all rows to Numbers. Ensure that all rows contains elements that can be converted to numerical data, othewise, elements that cannot be converted to int (like words) will be represented as NaN.

```Javascript
const formatted_dataset = csv.rowsToInt(dataset);
csv.tabularize(formatted_dataset); // or to view the changes, use console.log();
```

You may have unwanted columns in your current dataset (or columns that contains NaN values), considering dropping them using removeColumns() method. When removing columns, this will alter the structure of your dataset.

```Javascript
const cleaned_dataset = csv.removeColumns(["column_1", "column_2", "column_3"], formatted_dataset);
csv.tabularize(cleaned_dataset);
```

If you're going to feed your dataset to your model, consider normalizing your dataset first. Your dataset must not contain NaNs and all rows are already formatted to numbers. You can do this using normalize() method. As for now, the only available normalization method is MinMax, which normalize all values between 0 to 1.

```Javascript
const normalized_dataset = csv.normalize("MinMax",cleaned_dataset);
csv.tabularize(normalized_dataset);
```

If you want to extract data under the column, use extractColumn(). Note that this will alter the structure of your dataset.

```Javascript
const extracted_column = csv.extractColumn("column_1", cleaned_dataset);
csv.tabularize(extracted_column);
```

### To Do
- now working to support training on GPU.

## Notes
 - New version (version 0.0.6) is underway.

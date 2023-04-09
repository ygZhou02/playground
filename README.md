# Deep playground plus

We add functions on [tensorflow playground project](https://github.com/tensorflow/playground), including various normalizations, optimizers and initializations. Below are brief introduction about how we build up these functions.

## Interface

Firstly we should update the original interface to demonstrate controlling boxes of below functions.
This can be modified in [index.html](index.html) via adding controlling UI classes and assign value to controlling variable.
For example, in order to create a new activation function leaky ReLU, we firstly declare it in [index.html](index.html),
and then add a new option in controlling string `activations` in [state.ts](src/state.ts). 
In this way, we establish a data connection between User Interface and typescript code. 
If the user choose `leaky ReLU` in `activations` box, this change will inform the ts code that the network state is changed and the whole network will be re-built.

## Data flow

Data flow includes training mode and inference mode, and the latter is responsible for the fantastic decision boundary visualization. However, in this chapter we focus on the data flow in neural network training. The main code file concerning network training is [nn.ts](src/nn.ts).

First, training and testing dataset is created in [dataset.ts](src/dataset.ts). Via random choice, training data is selected in function oneStep() in [playground.ts](src/playground.ts) and passed to network training functions.
In order to achieve batch-based operation, we modify the data flow to a "batch-based" framework--all the data pass in and out every training function is a batch of data.
This modification adds complexity to our data processing in each function, but this is a crucial step to achieve batch normalization.

The training data then follow the sequence of forward propagation, back propagation and weight update. 
- forwardProp() revokes node function to propagate the input data up to the bottom of the network.
- backProp() calculates the derivative of the loss function, and back-propagate this derivative to the first layer and links.
- updateWeights() updates the weights in links and the biases in nodes based on optimizer and its learning rate.

## Normalizations

Following the interface part, we create a controlling variable `normalization` in [state.ts](src/state.ts).
This variable includes 3 values: 0 for no normalization, 1 for batch normalization, and 2 for layer normalization. If this controlling variable is changed, the whole network will be reformed and a new kind of normalization layer will be inserted in the network.

We create an interface NormalizationLayer, which contains two implements: BatchNormalization and LayerNormalization.


### Layer normalization

A layer normalization layer take an input vector of shape (D, N), where D stands for the number of nodes in a layer, N stands for batch size.
In the forward function, this layer normalizes the input vector in the first dimension and implement a neuron by neuron affine transformation; in the backward function, this layer calculates the derivatives of affine layer, and then passes its derivative to former layer.

TODO:加入原理分析，公式排上去

$$ a_2 $$


Detailed implementation can be found in class LayerNormalization in [nn.ts](src/nn.ts).



### Batch normalization



## Optimizer

We build 2 classic optimizer: SGD and Adam (to be continued).

### Stochastic gradient descent (SGD)

Based on random selected batch, SGD can be easily built. 

### Adam

Optimizer

Below follows the original [readme.md](https://github.com/tensorflow/playground/blob/master/README.md) in tensorflow playground.

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`.

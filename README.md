# Deep playground plus

We add functions on [tensorflow playground project](https://github.com/tensorflow/playground), including various normalizations, optimizers and initializations. Below are brief introduction about how we build up these functions.

## Interface

Firstly we should update the original interface to demonstrate controlling boxes of below functions.
This can be modified in [index.html](index.html) via adding controlling UI classes and assign value to controlling variable.
For example, in order to create a new activation function leaky ReLU, we firstly declare it in [index.html](index.html),
and then add a new option in controlling string `activations` in [state.ts](src/state.ts). 
In this way, we establish a data connection between User Interface and typescript code. 
If the user choose `leaky ReLU` in `activations` box, this change will inform the ts code that the network state is changed and the whole network will be re-built.

## Normalizations

Following the interface part, we create a controlling variable `normalization` in [state.ts](src/state.ts).

### Layer normalization

### Batch normalization

## Optimizer

### Stochastic gradient descent (SGD)

### Adam

Optimizer

Below part follows the original [readme.md](https://github.com/tensorflow/playground/blob/master/README.md) in tensorflow playground.

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

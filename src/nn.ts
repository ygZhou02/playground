/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  /** List of input links. */
  inputLinks: Link[] = [];
  bias = 0.1;
  /** List of output links. */
  outputs: Link[] = [];
  totalInput: number[];
  output: number[];
  /** Error derivative with respect to this node's output. */
  outputDer: number[] = [];
  /** Error derivative with respect to this node's total input. */
  inputDer: number[] = [];
  /**
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  accInputDer:number = 0;
  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  numAccumulatedDers = 0;
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;
  /** Which normalization to use, 0--none, 1--batch norm, 2--layer norm */
  normalization: number;
  /** Biased first moment estimate of bias. */
  m_t = 0;
  /** Biased second raw moment estimate of bias. */
  v_t = 0;
  /** Normalization layer */
  normlayer: NormalizationLayer;

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, normalization: number, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.normalization = normalization;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number[] {
    // Stores total input into the node.
    let batch_size = this.inputLinks[0].source.output.length;
    this.totalInput = new Array(batch_size);
    this.output = new Array(batch_size);
    for(let i=0; i<batch_size; i++){
      this.totalInput[i] = this.bias;
      for (let j = 0; j < this.inputLinks.length; j++) {
        let link = this.inputLinks[j];
        if (this.normalization !== 0 && link.source.normlayer !== undefined) {
          this.totalInput[i] += link.weight * link.source.normlayer.output[j][i];
        }
        else{
          this.totalInput[i] += link.weight * link.source.output[i];
        }
      }
      this.output[i] = this.activation.output(this.totalInput[i]);
    }
    return this.output;
  }
}

/** Build-in optimization functions */
class Optimizer {
  private static SGD(param: any, lr: number){
    if (param instanceof Node)
      param.bias -= lr * param.accInputDer / param.numAccumulatedDers;
    else
      param.weight -= lr * param.accErrorDer / param.numAccumulatedDers;
  }

  private static Adam(param: any, lr: number, iter: number){
    let beta1 = 0.9;
    let beta2 = 0.999;
    let g: number;
    let m_hat: number;
    let v_hat: number;

    if (param instanceof Node)
      g = param.accInputDer / param.numAccumulatedDers;
    else
      g = param.accErrorDer / param.numAccumulatedDers;
    param.m_t = beta1 * param.m_t + (1 - beta1) * g;
    param.v_t = beta2 * param.v_t + (1 - beta2) * g * g;
    m_hat = param.m_t / (1 - Math.pow(beta1, iter));
    v_hat = param.v_t / (1 - Math.pow(beta2, iter));
    if (param instanceof Node)
      param.bias -= lr * m_hat / (Math.pow(v_hat, 0.5) + 1e-8);
    else
      param.weight -= lr * m_hat / (Math.pow(v_hat, 0.5) + 1e-8);
  }

  public static step(optimizer_id: number, param: any, lr: number, iter: number){
    if (optimizer_id == 0)
      this.SGD(param, lr)
    else
      this.Adam(param, lr, iter)
  }
}

function deepCopy(obj) {
    var copy;

    // Handle the 3 simple types, and null or undefined
    if (null == obj || "object" != typeof obj) return obj;

    // Handle Date
    if (obj instanceof Date) {
        copy = new Date();
        copy.setTime(obj.getTime());
        return copy;
    }

    // Handle Array
    if (obj instanceof Array) {
        copy = [];
        for (var i = 0, len = obj.length; i < len; i++) {
            copy[i] = deepCopy(obj[i]);
        }
        return copy;
    }

    // Handle Object
    if (obj instanceof Object) {
        copy = {};
        for (var attr in obj) {
            if (obj.hasOwnProperty(attr)) copy[attr] = deepCopy(obj[attr]);
        }
        return copy;
    }

    throw new Error("Unable to copy obj! Its type isn't supported.");
}


interface NormalizationLayer{
  gamma: number[];
  beta: number[];
  variation: number[];
  input: number[][];
  output: number[][];
  dgamma: number[];
  dbeta: number[];
  dX: number[][];

  forward(X: number[][], mode: string): number[][];
  backward(dY: number[][]): number[][];
}


export class BatchNormalization implements NormalizationLayer{
  moving_mean: number[];
  moving_var: number[];
  mean: number[];  // batch mean
  variation: number[];  // batch variation
  gamma: number[];
  beta: number[];
  eps: number;
  decay: number;
  Xhat: number[][];
  input: number[][];
  output: number[][];

  dgamma: number[];
  dbeta: number[];
  dX: number[][];

  constructor(width: number){
    this.moving_mean = new Array(width);
    this.moving_var = new Array(width);
    this.gamma = new Array(width);
    this.beta = new Array(width);
    this.eps = 1e-5;
    this.decay = 0.95;

    for(let i=0; i<width; i++){
      this.moving_mean[i] = 0;
      this.moving_var[i] = 0;
      this.gamma[i] = 1;
      this.beta[i] = 0;
    }
  }

  forward(X: number[][], mode: string): number[][] {
    this.input = deepCopy(X);
    this.output = deepCopy(X);
    let Xhat = deepCopy(X);
    let L = X.length;  // layer size
    let N = X[0].length;  // batch size
    let mean = new Array(L);
    let variation = new Array(L);
    if(mode === 'eval'){
      for(let i=0; i<L; i++){
        for(let j=0; j<N; j++){
          Xhat[i][j] = (X[i][j] - this.moving_mean[i]) / Math.sqrt(this.moving_var[i]+this.eps);
          this.output[i][j] = this.gamma[i] * Xhat[i][j] + this.beta[i];
        }
      }
      return this.output
    }
    for(let i=0; i<L; i++){
      // calculate mean and variation
      mean[i] = 0;
      variation[i] = 0;
      for(let j=0; j<N; j++){
        mean[i] += X[i][j];
      }
      mean[i] /= N;
      for(let j=0; j<N; j++){
        variation[i] += (X[i][j] - mean[i]) * (X[i][j] - mean[i]);
      }
      variation[i] = variation[i] / N;
      // normalization
      for(let j=0; j<N; j++){
        Xhat[i][j] = (X[i][j] - mean[i]) / Math.sqrt(variation[i]+this.eps);
        this.output[i][j] = this.gamma[i] * Xhat[i][j] + this.beta[i];
      }
      this.moving_mean[i] = this.decay*this.moving_mean[i] + (1-this.decay)*mean[i];
      this.moving_var[i] = this.decay*this.moving_var[i] + (1-this.decay)*variation[i];
    }
    this.Xhat = deepCopy(Xhat);
    this.mean = deepCopy(mean);
    this.variation = deepCopy(variation);
    return this.output;
  }
  backward(dY: number[][]): number[][]{
    let L = dY.length;
    let N = dY[0].length;
    this.dX = deepCopy(dY);
    this.dgamma = new Array(N);
    this.dbeta = new Array(N);
    for(let i=0; i<L; i++){
      this.dbeta[i] = 0;
      this.dgamma[i] = 0;
      for(let j=0; j<N; j++){
        this.dbeta[i] += dY[i][j];
        this.dgamma[i] += dY[i][j] * this.Xhat[i][j];
      }
    }
    for(let i=0; i<L; i++){
      let dvar = 0;
      let dmean = 0;
      let k = Math.sqrt((this.variation[i]+this.eps)*(this.variation[i]+this.eps)*(this.variation[i]+this.eps));
      for(let j=0; j<N; j++){
        dvar += dY[i][j]*this.gamma[i]*(this.input[i][j]-this.mean[i])*-0.5/k;
        dmean += -dY[i][j]*this.gamma[i] / Math.sqrt(this.variation[i]+this.eps)
      }
      for(let j=0; j<N; j++){
        this.dX[i][j] += dY[i][j]*this.gamma[i]/Math.sqrt(this.variation[i]+this.eps)+dvar*2*(this.input[i][j]-this.mean[i])/N+dmean/N;
      }
    }
    return this.dX;
  }
}


export class LayerNormalization implements NormalizationLayer{

  gamma: number[];
  beta: number[];
  eps: number;
  input: number[][];
  Xhat: number[][];
  output: number[][];
  variation: number[];
  dgamma: number[];
  dbeta: number[];
  dX: number[][];

  constructor(width: number){
    this.gamma = new Array(width);
    for(let i=0; i<width; i++){
      this.gamma[i] = 1;
    }
    this.beta = new Array(width);
    for(let i=0; i<width; i++){
      this.beta[i] = 0;
    }
    this.eps = 1e-5;
  }

  /**
   *     Input:
   *     - X: Data of shape (D, N)
   *     - gamma: Scale parameter of shape (D,)
   *     - beta: Shift paremeter of shape (D,)
   *     - ln_param: Dictionary with the following keys:
   *         - eps: Constant for numeric stability
   */
  forward(X: number[][], mode: string): number[][] {
    this.input = deepCopy(X);
    this.output = deepCopy(X);
    let D = X.length;
    let N = X[0].length;
    let mu = new Array(N);
    let variation = new Array(N);
    let Xhat = deepCopy(X);
    for(let i=0; i<N; i++){
      // calculate average and variation
      mu[i] = 0;
      variation[i] = 0;
      for(let j=0; j<D; j++){
        mu[i] += X[j][i];
      }
      mu[i] /= D;
      for(let j=0; j<D; j++){
        variation[i] += (X[j][i] - mu[i]) * (X[j][i] - mu[i]);
      }
      variation[i] = Math.sqrt(variation[i] / D + this.eps);
      // normalization
      for(let j=0; j<D; j++){
        Xhat[j][i] = (X[j][i] - mu[i]) / variation[i];
        this.output[j][i] = this.gamma[j] * Xhat[j][i] + this.beta[j];
      }
    }
    this.Xhat = deepCopy(Xhat);
    this.variation = deepCopy(variation);
    return this.output;
  }

  // layer normalization back propagation
  backward(dY: number[][]): number[][]{
    let D = dY.length;
    let N = dY[0].length;
    this.dX = deepCopy(dY);
    this.dgamma = new Array(D);
    this.dbeta = new Array(D);
    for(let j=0; j<D; j++){
      this.dbeta[j] = 0;
      this.dgamma[j] = 0;
      for(let i=0; i<N; i++){
        this.dbeta[j] += dY[j][i];
        this.dgamma[j] += dY[j][i] * this.Xhat[j][i];
      }
    }
    for(let i=0; i<N; i++){
      let sumdXhat = 0;
      let sumdXhatXhat = 0;
      for(let k=0; k<D; k++){
        sumdXhat += (dY[k][i] * this.gamma[k]);
        sumdXhatXhat += (dY[k][i] * this.gamma[k] * this.Xhat[k][i]);
      }
      for(let j=0; j<D; j++){
        this.dX[j][i] = 1 / (D * this.variation[i]) * (D * dY[j][i] * this.gamma[j] -
                        this.Xhat[j][i] * sumdXhatXhat - sumdXhat);
      }
    }
    return this.dX;
  }
}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/** Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function(x) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** Built-in activation functions */
export class Activations {
  public static TANH: ActivationFunction = {
    output: x => (Math as any).tanh(x),
    der: x => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    }
  };
  public static RELU: ActivationFunction = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  };
  public static SIGMOID: ActivationFunction = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    }
  };
  public static LINEAR: ActivationFunction = {
    output: x => x,
    der: x => 1
  };
  public static LEAKYRELU: ActivationFunction = {
    output: x => x <= 0 ? 0.01 * x : x,
    der: x => x <= 0 ? 0.01 : 1
  };
}

/** Build-in regularization functions */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: w => Math.abs(w),
    der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
  };
  public static L2: RegularizationFunction = {
    output: w => 0.5 * w * w,
    der: w => w
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5;
  isDead = false;
  /** Error derivative with respect to this weight. */
  errorDer: number[] = [];
  /** Accumulated error derivative since the last update. */
  accErrorDer = 0;
  /** Number of accumulated derivatives since the last update. */
  numAccumulatedDers = 0;
  regularization: RegularizationFunction;
  /** Biased first moment estimate of bias. */
  m_t = 0;
  /** Biased second raw moment estimate of bias. */
  v_t = 0;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(source: Node, dest: Node,
      regularization: RegularizationFunction, weight: number, initZero?: number) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero === 0) {
      this.weight = 0;
    }
    else if(initZero === 1) {
      //Xavier Uniform
      this.weight = weight;
    }
  }
}

/**
 * transpose a matrix
 * @param matrix the matrix to be transposed
 */
export function transpose(matrix: number[][]) : number[][] {
  let D = matrix.length;
  let N = matrix[0].length;
  let output : number[][] = [];
  for(let i=0; i<N; i++){
    let vector = new Array(D);
    for(let j=0; j<D; j++){
      vector[j] = matrix[j][i];
    }
    output.push(vector);
  }
  return output;
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
    networkShape: number[], normalization: number, initialization: number, activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: number): Node[][] {
  let numLayers = networkShape.length;
  console.log(networkShape)
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    let normlayer: NormalizationLayer;
    if (normalization === 1 && !isInputLayer && !isOutputLayer) {
      normlayer = new BatchNormalization(numNodes);
    }
    if (normalization === 2 && !isInputLayer && !isOutputLayer) {
      normlayer = new LayerNormalization(numNodes);
    }
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId, normalization,
          isOutputLayer ? outputActivation : activation, true);
      // Add the same layer norm to all nodes in one layer.
      if (normalization !== 0 && !isInputLayer && !isOutputLayer) {
        node.normlayer = normlayer;
      }
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let weight = 0;
          if (initialization === 1) {
            weight = Math.random() - 0.5
          }
          else if (initialization === 2) {
            // Xavier Uniform
            if (activation === Activations.SIGMOID) {
              weight = 2 * (Math.random() - 0.5) * (Math.sqrt(6 / (networkShape[layerIdx - 1] + networkShape[layerIdx])));
            }
            else if (activation === Activations.RELU || activation === Activations.LEAKYRELU || activation === Activations.LINEAR) {
              weight = 2 * (Math.random() - 0.5) * (Math.sqrt(6 * 2 / (networkShape[layerIdx - 1] + networkShape[layerIdx])));
            }
            else if (activation === Activations.TANH) {
              weight = 2 * (Math.random() - 0.5) * (Math.sqrt(6 * 16 / (networkShape[layerIdx - 1] + networkShape[layerIdx])));
            }
          }
          // Xavier Gaussian is too hard! Uniform should be transformed to Gaussian, and then transformed to Gaussian with another variance!
          else if (initialization === 3) {
            // Kaiming Uniform
            if (activation === Activations.SIGMOID) {
              weight = 2 * (Math.random() - 0.5) * (Math.sqrt(6 / (networkShape[layerIdx - 1])));
            }
            else if (activation === Activations.RELU || activation === Activations.LEAKYRELU || activation === Activations.LINEAR) {
              weight = 2 * (Math.random() - 0.5) * (Math.sqrt(6 * 2 / (networkShape[layerIdx - 1])));
            }
            else if (activation === Activations.TANH) {
              weight = 2 * (Math.random() - 0.5) * (Math.sqrt(6 * 16 / (networkShape[layerIdx - 1])));
            }
          }
          let link = new Link(prevNode, node, regularization, weight, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param batch The input array. Its length should match the number of input
 *     nodes in the network.
 * @param mode Set for Batch Normalization. Choose from 'train' and 'eval'.
 * @return The final output of the network.
 */
export function forwardProp(network: Node[][], batch: number[][], mode: string): number[] {
  let inputLayer = network[0];
  if (batch[0].length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    let dim1 = [];
    for(let j=0; j<batch.length; j++){
      dim1.push(batch[j][i])
    }
    node.output = dim1;
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    let outputList : number[][] = [];
    let normlayer = currentLayer[0].normlayer;
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let vector = node.updateOutput();
      if (node.normalization !== 0 && layerIdx !== network.length - 1){
        outputList.push(vector);
      }
    }
    if (outputList.length > 0 && layerIdx !== network.length - 1){
      normlayer.forward(outputList, mode);
    }
  }
  return network[network.length - 1][0].output;
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: Node[][], target: number[],
    errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  let outputNode = network[network.length - 1][0];
  let batch_size = target.length;
  outputNode.outputDer = []
  // accumulate all derivatives
  for(let i=0; i<batch_size; i++){
    outputNode.outputDer.push(errorFunc.der(outputNode.output[i], target[i]));
  }
  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    // Normalization layer backward
    // if (currentLayer[0].normalization === 2 && currentLayer[0].normlayer !== null){
    if (currentLayer[0].normalization !== 0 && currentLayer[0].normlayer !== undefined){
      let normlayer = currentLayer[0].normlayer;
      let outputDer: number[][] = [];
      for(let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        let vector = deepCopy(node.outputDer);
        outputDer.push(vector);
      }
      normlayer.backward(outputDer);
      for(let i = 0; i < currentLayer.length; i++) {
        let node = currentLayer[i];
        // let vector = deepCopy(normlayer.dX[i]);
        node.outputDer = deepCopy(normlayer.dX[i]);
      }
    }
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // accumulate all derivatives
      node.inputDer = new Array(batch_size);
      node.numAccumulatedDers = batch_size;
      node.accInputDer = 0;
      for(let j=0; j<node.totalInput.length; j++){
        node.inputDer[j] = node.outputDer[j] * node.activation.der(node.totalInput[j]);
        node.accInputDer += node.inputDer[j];
      }
    }

    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          console.log("link is dead!!!")
          continue;
        }
        link.numAccumulatedDers = batch_size;
        link.accErrorDer = 0;
        for(let k=0; k<batch_size; k++){
          link.errorDer[k] = node.inputDer[k] * link.source.output[k];
          link.accErrorDer += link.errorDer[k];
        }
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    let prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      node.outputDer = new Array(batch_size);
      for (let k=0; k<batch_size; k++){
        node.outputDer[k] = 0;
        for (let j = 0; j < node.outputs.length; j++) {
          let output = node.outputs[j];
          node.outputDer[k] += output.weight * output.dest.inputDer[k];
        }
      }
    }
  }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
export function updateWeights(network: Node[][], learningRate: number,
    regularizationRate: number, optimization: number, iter: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    let normlayer = currentLayer[0].normlayer;
    if(currentLayer[0].normalization !== 0 && normlayer !== undefined){
      for(let i=0; i<normlayer.gamma.length; i++){
        normlayer.gamma[i] = normlayer.gamma[i] - learningRate * normlayer.dgamma[i];
        normlayer.beta[i] = normlayer.beta[i] - learningRate * normlayer.dbeta[i];
      }
    }
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        //node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
        Optimizer.step(optimization, node, learningRate, iter);
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }
      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        let regulDer = link.regularization ?
            link.regularization.der(link.weight) : 0;
        if (link.numAccumulatedDers > 0) {
          // Update the weight based on dE/dw.
          Optimizer.step(optimization, link, learningRate, iter);
          // Further update the weight based on regularization.
          let newLinkWeight = link.weight -
              (learningRate * regularizationRate) * regulDer;
          if (link.regularization === RegularizationFunction.L1 &&
              link.weight * newLinkWeight < 0) {
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}

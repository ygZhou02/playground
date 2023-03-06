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
  totalInput: number[];     //x_i;
  totalInputHat: number[];     //x^_i
  totalInputHatAfterGamma: number[];     //y^_i
  output: number[];         // Activate(y_i)
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
  /** batch normalization affine factor*/
  gamma: number;
  dgamma: number;
  beta: number;
  dbeta: number;
  running_mean = 0.0;
  running_var = 1.0;
  momentum: number;
  cache = [];

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
    if(this.normalization === 1){
      this.gamma = 1.0;
      this.beta = 0.0;
      this.running_mean = 0.0;
      this.running_var = 1.0;
      this.momentum = 0.9;
    }
    this.cache = [];
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number[] {
    // Stores total input into the node.
    // This function will never act on the first layer.
    let batch_size = this.inputLinks[0].source.output.length;
    this.totalInput = new Array(batch_size);
    this.output = new Array(batch_size);
    for(let i=0; i<batch_size; i++){
      this.totalInput[i] = this.bias;
      for (let j = 0; j < this.inputLinks.length; j++) {
        let link = this.inputLinks[j];
        this.totalInput[i] += link.weight * link.source.output[i];
      }
    }
    let sample_mean = 0;
    let sample_var = 0;
    if(this.normalization === 1) {
      sample_mean = getMean(this.totalInput);
      // Unbiased estimation
      sample_var = getVar(this.totalInput, sample_mean);
      // Only update running mean and running var on training mode.
      if(batch_size > 1){
        // Due to the huge computational cost to compute the average and mean of whole training dataset,
        // we estimate mean and variance by momentum mechanism.
        this.running_mean = this.momentum * this.running_mean + (1 - this.momentum) * sample_mean;
        this.running_var = this.momentum * this.running_var + (1 - this.momentum) * batch_size / Math.max(batch_size - 1, 1) * sample_var;
      }
    }

    if(this.normalization === 1) {
      this.totalInputHat = new Array(batch_size);
      this.totalInputHatAfterGamma = new Array(batch_size);
    }

    // calculate outputs by sample.
    for(let i=0; i<batch_size; i++){
      if(this.normalization === 1) {
        // execute batch normalization
        if(batch_size === 1){
          // Inference mode
          this.totalInputHat[i] = (this.totalInput[i] - this.running_mean) / (Math.sqrt(1e-5 + this.running_var))
          this.totalInputHatAfterGamma[i] = this.gamma * this.totalInputHat[i] + this.beta;
          this.output[i] = this.activation.output(this.totalInputHatAfterGamma[i]);
        }
        else{
          // batch size > 1 means training mode
          this.totalInputHat[i] = (this.totalInput[i] - sample_mean) / (Math.sqrt(1e-5 + sample_var))
          this.totalInputHatAfterGamma[i] = this.gamma * this.totalInputHat[i] + this.beta;
          this.output[i] = this.activation.output(this.totalInputHatAfterGamma[i]);
          this.cache = [this.totalInputHat, sample_mean, sample_var];
        }
      }
      else if(this.normalization === 0){
        // no normalization
        this.output[i] = this.activation.output(this.totalInput[i]);
      }
    }
    return this.output;
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

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(source: Node, dest: Node,
      regularization: RegularizationFunction, initZero?: boolean) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }
}


export function getMean(batch: number[]){
  let sum = 0;
  for(let i=0; i<batch.length; i++){
    sum += batch[i];
  }
  return sum / batch.length;
}

export function getVar(batch: number[], mean: number){
  let sum = 0;
  for(let i=0; i<batch.length; i++){
    sum += (batch[i] - mean) * (batch[i] - mean);
  }
  // Biased estimate.
  return sum / batch.length;
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
    networkShape: number[], normalization: number, activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: boolean): Node[][] {
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
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId, normalization,
          isOutputLayer ? outputActivation : activation, initZero);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
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
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network.
 */
export function forwardProp(network: Node[][], batch: number[][]): number[] {
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
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.updateOutput();
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
  outputNode.outputDer = []    // outputDer = partial loss / partial Activate(y_i)
  // accumulate all derivatives
  outputNode.outputDer.push(errorFunc.der(outputNode.output[0], target[0]));
  for(let i=1; i<batch_size; i++){
    outputNode.outputDer.push(errorFunc.der(outputNode.output[i], target[i]));
  }
  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // accumulate all derivatives
      node.inputDer = new Array(batch_size);   // inputDer = partial loss / partial y_i (x_i)
      node.numAccumulatedDers = batch_size;
      node.accInputDer = 0;     // accInputDer = partial loss / partial b
      for(let j=0; j<batch_size; j++){
        // node.inputDer[i] = partial loss / partial y_i
        node.inputDer[j] = node.outputDer[j] * node.activation.der(node.totalInput[j]);
        node.accInputDer += node.inputDer[j];     // accInputDer = partial loss / partial b
      }

      if(node.normalization === 1){
        // batch normalization backpropagation
        let xhat = node.cache[0];
        let gamma = node.gamma;
        let mean = node.cache[1];
        let variance = node.cache[2];
        // xmu = x - u_B
        let xmu = new Array(batch_size);
        // dxhat = partial loss / partial x^
        let dxhat = new Array(batch_size);
        // dvar = partial loss / partial variance
        let dvar = 0;
        for(let k=0; k<batch_size; k++){
          xmu[k] = node.totalInput[k] - mean;
          dxhat[k] = node.inputDer[k] * gamma;
          dvar += dxhat[k] * xmu[k];
        }
        dvar *= (-0.5 / Math.sqrt(variance + 1e-5) / (variance + 1e-5));
        // dmean = partial loss / partial u_B
        let dmean = 0;
        for(let k=0; k<batch_size; k++){
          dmean += (-dxhat[k] / Math.sqrt(variance + 1e-5)); // + dvar / batch_size * (-2) * xmu[k]);
        }
        node.dbeta = node.accInputDer;
        node.dgamma = 0;
        for(let k=0; k<batch_size; k++){
          node.dgamma += node.inputDer[k] * xhat[k];
        }
        node.accInputDer = 0;
        for(let k=0; k<batch_size; k++){
          // node.inputDer[i] = partial loss / partial x_i
          node.inputDer[k] = dxhat[k] / Math.sqrt(variance + 1e-5) + 2.0 * dvar * xmu[k] / batch_size + dmean / batch_size;
          node.accInputDer += node.inputDer[k];
        }
      }
    }

    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          console.log("The link is dead!!!")
          continue;
        }
        link.numAccumulatedDers = batch_size;
        link.accErrorDer = 0;     // accErrorDer = partial loss / partial W
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
      node.outputDer = new Array(outputNode.outputDer.length);   // partial loss / partial in_layer
      for (let k=0; k<outputNode.outputDer.length; k++){
        node.outputDer[k] = 0
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
    regularizationRate: number) {

  //TODO: rectify backpropagation and gradient update!
  // why the model will break down upon it reaches the best answer?
  // observe how the model behaves near the local minimum.


  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
        node.accInputDer = 0;
        if(node.normalization === 1){
          node.gamma -= learningRate * node.dgamma / node.numAccumulatedDers;
          node.beta -= learningRate * node.dbeta / node.numAccumulatedDers;
          node.dbeta = 0;
          node.dgamma = 0;
        }
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
          link.weight = link.weight -
              (learningRate / link.numAccumulatedDers) * link.accErrorDer;
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

// serialize.js — Model serialization (save/load weights as JSON)
//
// Saves network architecture and weights to a portable format.

import { Matrix } from './matrix.js';

/**
 * Serialize a network's weights to a JSON-compatible object
 */
export function serializeWeights(network) {
  const layers = [];
  for (const layer of network.layers) {
    const layerData = { type: layer.constructor.name };
    
    if (layer.weights) {
      layerData.weights = matrixToJSON(layer.weights);
    }
    if (layer.biases) {
      layerData.biases = matrixToJSON(layer.biases);
    }
    if (layer.gamma) {
      layerData.gamma = matrixToJSON(layer.gamma);
    }
    if (layer.beta) {
      layerData.beta = matrixToJSON(layer.beta);
    }
    if (layer.runningMean) {
      layerData.runningMean = matrixToJSON(layer.runningMean);
    }
    if (layer.runningVar) {
      layerData.runningVar = matrixToJSON(layer.runningVar);
    }
    
    // Save layer config
    if (layer.inputSize !== undefined) layerData.inputSize = layer.inputSize;
    if (layer.outputSize !== undefined) layerData.outputSize = layer.outputSize;
    if (layer.activation) layerData.activation = layer.activation;
    if (layer.rate !== undefined) layerData.rate = layer.rate;
    
    layers.push(layerData);
  }
  
  return { layers, version: 1 };
}

/**
 * Deserialize weights into a network
 */
export function deserializeWeights(network, data) {
  if (data.version !== 1) {
    throw new Error(`Unknown serialization version: ${data.version}`);
  }
  
  if (data.layers.length !== network.layers.length) {
    throw new Error(`Layer count mismatch: expected ${network.layers.length}, got ${data.layers.length}`);
  }
  
  for (let i = 0; i < data.layers.length; i++) {
    const layer = network.layers[i];
    const layerData = data.layers[i];
    
    if (layerData.weights && layer.weights) {
      layer.weights = jsonToMatrix(layerData.weights);
    }
    if (layerData.biases && layer.biases) {
      layer.biases = jsonToMatrix(layerData.biases);
    }
    if (layerData.gamma && layer.gamma) {
      layer.gamma = jsonToMatrix(layerData.gamma);
    }
    if (layerData.beta && layer.beta) {
      layer.beta = jsonToMatrix(layerData.beta);
    }
    if (layerData.runningMean && layer.runningMean) {
      layer.runningMean = jsonToMatrix(layerData.runningMean);
    }
    if (layerData.runningVar && layer.runningVar) {
      layer.runningVar = jsonToMatrix(layerData.runningVar);
    }
  }
}

/**
 * Convert Matrix to JSON-serializable format
 */
function matrixToJSON(matrix) {
  return {
    rows: matrix.rows,
    cols: matrix.cols,
    data: Array.from(matrix.data),
  };
}

/**
 * Restore Matrix from JSON format
 */
function jsonToMatrix(json) {
  const m = new Matrix(json.rows, json.cols);
  for (let i = 0; i < json.data.length; i++) {
    m.data[i] = json.data[i];
  }
  return m;
}

/**
 * Convert network to JSON string
 */
export function saveToJSON(network) {
  return JSON.stringify(serializeWeights(network));
}

/**
 * Load weights from JSON string
 */
export function loadFromJSON(network, jsonString) {
  deserializeWeights(network, JSON.parse(jsonString));
}

/**
 * Compute a checksum of the weights (for comparison)
 */
export function weightsChecksum(network) {
  let sum = 0;
  for (const layer of network.layers) {
    if (layer.weights) {
      for (let i = 0; i < layer.weights.data.length; i++) {
        sum += layer.weights.data[i];
      }
    }
  }
  return sum;
}

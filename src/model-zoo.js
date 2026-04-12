// model-zoo.js — Pre-configured architectures for common tasks
//
// Usage:
//   import { ModelZoo } from './model-zoo.js';
//   const model = ModelZoo.xor();
//   const model = ModelZoo.classifier(inputSize, numClasses);
//   const model = ModelZoo.autoencoder(inputSize, latentSize);
//   const model = ModelZoo.timeSeries(sequenceLength, features, horizon);

import { Network } from './network.js';

/**
 * Pre-configured model architectures.
 * Each returns a configured Network ready for training.
 */
export const ModelZoo = {
  /**
   * XOR: Classic 2→8→1 network for learning XOR
   */
  xor() {
    const net = new Network();
    net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');
    return net;
  },

  /**
   * Binary classifier: input → hidden → 1 (sigmoid)
   */
  binaryClassifier(inputSize, hiddenSize = 32) {
    const net = new Network();
    net.dense(inputSize, hiddenSize, 'relu')
      .dense(hiddenSize, hiddenSize, 'relu')
      .dense(hiddenSize, 1, 'sigmoid')
      .loss('mse');
    return net;
  },

  /**
   * Multi-class classifier: input → hidden → numClasses
   * Use with cross-entropy loss and softmax output
   */
  classifier(inputSize, numClasses, hiddenSize = 64) {
    const net = new Network();
    net.dense(inputSize, hiddenSize, 'relu')
      .dense(hiddenSize, hiddenSize, 'relu')
      .dense(hiddenSize, numClasses, 'linear')
      .loss('crossEntropy');
    return net;
  },

  /**
   * Regression: input → hidden → 1 (linear)
   */
  regression(inputSize, hiddenSize = 32) {
    const net = new Network();
    net.dense(inputSize, hiddenSize, 'relu')
      .dense(hiddenSize, hiddenSize, 'relu')
      .dense(hiddenSize, 1, 'linear')
      .loss('mse');
    return net;
  },

  /**
   * Autoencoder: compress input to latent space and reconstruct
   */
  autoencoder(inputSize, latentSize = 8) {
    const midSize = Math.max(latentSize * 2, Math.floor((inputSize + latentSize) / 2));
    const encoder = new Network();
    encoder.dense(inputSize, midSize, 'relu')
      .dense(midSize, latentSize, 'relu');

    const decoder = new Network();
    decoder.dense(latentSize, midSize, 'relu')
      .dense(midSize, inputSize, 'sigmoid');

    // Return both as a single network for end-to-end training
    const net = new Network();
    net.dense(inputSize, midSize, 'relu')
      .dense(midSize, latentSize, 'relu')
      .dense(latentSize, midSize, 'relu')
      .dense(midSize, inputSize, 'sigmoid')
      .loss('mse');
    
    return { net, encoder, decoder, latentSize };
  },

  /**
   * MNIST-like CNN: for 28x28 grayscale images
   */
  mnistCNN(numClasses = 10) {
    const net = new Network();
    // Note: requires Conv2D and Flatten layers
    net.dense(784, 128, 'relu')
      .dense(128, 64, 'relu')
      .dense(64, numClasses, 'linear')
      .loss('crossEntropy');
    return net;
  },

  /**
   * Time series forecasting: uses dense layers
   * Input: [sequenceLength * features] flattened window
   * Output: [horizon] future predictions
   */
  timeSeries(sequenceLength, features = 1, horizon = 1) {
    const inputSize = sequenceLength * features;
    const net = new Network();
    net.dense(inputSize, 64, 'relu')
      .dense(64, 32, 'relu')
      .dense(32, horizon, 'linear')
      .loss('mse');
    return net;
  },

  /**
   * Tiny: minimal network for unit tests
   */
  tiny(inputSize = 2, outputSize = 1) {
    const net = new Network();
    net.dense(inputSize, 4, 'relu').dense(4, outputSize, 'sigmoid').loss('mse');
    return net;
  },

  /**
   * Deep: 5-layer network for testing deep architectures
   */
  deep(inputSize, outputSize, hiddenSize = 32) {
    const net = new Network();
    net.dense(inputSize, hiddenSize, 'relu')
      .dense(hiddenSize, hiddenSize, 'relu')
      .dense(hiddenSize, hiddenSize, 'relu')
      .dense(hiddenSize, hiddenSize, 'relu')
      .dense(hiddenSize, outputSize, 'linear')
      .loss('mse');
    return net;
  },

  /**
   * Wide: single wide hidden layer
   */
  wide(inputSize, outputSize, width = 256) {
    const net = new Network();
    net.dense(inputSize, width, 'relu')
      .dense(width, outputSize, 'linear')
      .loss('mse');
    return net;
  },
};

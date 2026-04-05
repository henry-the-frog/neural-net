// mnist.js — MNIST dataset loader and utilities
// Supports IDX binary format parsing, embedded subsets, and data preparation

import { Matrix } from './matrix.js';

/**
 * Parse MNIST IDX file format (images or labels)
 * IDX format: magic number (4 bytes), dimensions, then data
 * Images: magic=2051, dims=[count, rows, cols], data=uint8 pixels
 * Labels: magic=2049, dims=[count], data=uint8 labels
 */
export function parseIDX(buffer) {
  const view = new DataView(buffer instanceof ArrayBuffer ? buffer : buffer.buffer);
  const magic = view.getUint32(0);
  const type = (magic >> 8) & 0xff; // data type byte
  const ndims = magic & 0xff;       // number of dimensions

  const dims = [];
  for (let i = 0; i < ndims; i++) {
    dims.push(view.getUint32(4 + i * 4));
  }

  const offset = 4 + ndims * 4;
  const totalElements = dims.reduce((a, b) => a * b, 1);
  const data = new Uint8Array(buffer, offset, totalElements);

  return { dims, data: new Uint8Array(data) }; // copy to avoid detached buffer issues
}

/**
 * Load MNIST from IDX buffers and prepare training data
 * @param {ArrayBuffer} imageBuffer - IDX file for images
 * @param {ArrayBuffer} labelBuffer - IDX file for labels
 * @param {Object} options
 * @param {number} options.limit - Max samples to load (default: all)
 * @param {boolean} options.normalize - Normalize pixels to 0-1 (default: true)
 * @param {boolean} options.oneHot - One-hot encode labels (default: true)
 * @param {boolean} options.flatten - Flatten 28×28 to 784 (default: true)
 * @returns {{ images: Matrix[], labels: Matrix[], rawLabels: number[] }}
 */
export function loadMNIST(imageBuffer, labelBuffer, options = {}) {
  const { limit = Infinity, normalize = true, oneHot = true, flatten = true } = options;

  const imgData = parseIDX(imageBuffer);
  const lblData = parseIDX(labelBuffer);

  const count = Math.min(imgData.dims[0], lblData.dims[0], limit);
  const rows = imgData.dims[1] || 28;
  const cols = imgData.dims[2] || 28;
  const pixelsPerImage = rows * cols;

  const images = [];
  const labels = [];
  const rawLabels = [];

  for (let i = 0; i < count; i++) {
    // Extract pixel data
    const pixels = new Float64Array(pixelsPerImage);
    for (let j = 0; j < pixelsPerImage; j++) {
      const val = imgData.data[i * pixelsPerImage + j];
      pixels[j] = normalize ? val / 255 : val;
    }

    if (flatten) {
      images.push(new Matrix(pixelsPerImage, 1, pixels));
    } else {
      // Keep as 2D: rows × cols (single channel)
      images.push(new Matrix(rows, cols, pixels));
    }

    // Label
    const label = lblData.data[i];
    rawLabels.push(label);

    if (oneHot) {
      const oh = new Float64Array(10);
      oh[label] = 1;
      labels.push(new Matrix(10, 1, oh));
    } else {
      labels.push(new Matrix(1, 1, new Float64Array([label])));
    }
  }

  return { images, labels, rawLabels, rows, cols };
}

/**
 * Create a small embedded MNIST-like dataset for demos
 * Hand-crafted 8×8 digit patterns (similar to sklearn's digits dataset)
 * Returns ready-to-use training data
 */
export function createMiniDigits(options = {}) {
  const { samplesPerDigit = 10, noise = 0.05 } = options;

  // 8×8 prototype patterns for digits 0-9
  const prototypes = [
    // 0
    [0,0,1,1,1,1,0,0,
     0,1,0,0,0,0,1,0,
     1,0,0,0,0,0,0,1,
     1,0,0,0,0,0,0,1,
     1,0,0,0,0,0,0,1,
     1,0,0,0,0,0,0,1,
     0,1,0,0,0,0,1,0,
     0,0,1,1,1,1,0,0],
    // 1
    [0,0,0,1,1,0,0,0,
     0,0,1,1,1,0,0,0,
     0,0,0,1,1,0,0,0,
     0,0,0,1,1,0,0,0,
     0,0,0,1,1,0,0,0,
     0,0,0,1,1,0,0,0,
     0,0,0,1,1,0,0,0,
     0,0,1,1,1,1,0,0],
    // 2
    [0,0,1,1,1,1,0,0,
     0,1,0,0,0,0,1,0,
     0,0,0,0,0,0,1,0,
     0,0,0,0,0,1,0,0,
     0,0,0,0,1,0,0,0,
     0,0,0,1,0,0,0,0,
     0,0,1,0,0,0,0,0,
     0,1,1,1,1,1,1,0],
    // 3
    [0,0,1,1,1,1,0,0,
     0,1,0,0,0,0,1,0,
     0,0,0,0,0,0,1,0,
     0,0,0,1,1,1,0,0,
     0,0,0,0,0,0,1,0,
     0,0,0,0,0,0,1,0,
     0,1,0,0,0,0,1,0,
     0,0,1,1,1,1,0,0],
    // 4
    [0,0,0,0,0,1,0,0,
     0,0,0,0,1,1,0,0,
     0,0,0,1,0,1,0,0,
     0,0,1,0,0,1,0,0,
     0,1,0,0,0,1,0,0,
     1,1,1,1,1,1,1,0,
     0,0,0,0,0,1,0,0,
     0,0,0,0,0,1,0,0],
    // 5
    [0,1,1,1,1,1,1,0,
     0,1,0,0,0,0,0,0,
     0,1,0,0,0,0,0,0,
     0,1,1,1,1,1,0,0,
     0,0,0,0,0,0,1,0,
     0,0,0,0,0,0,1,0,
     0,1,0,0,0,0,1,0,
     0,0,1,1,1,1,0,0],
    // 6
    [0,0,1,1,1,1,0,0,
     0,1,0,0,0,0,0,0,
     1,0,0,0,0,0,0,0,
     1,0,1,1,1,1,0,0,
     1,1,0,0,0,0,1,0,
     1,0,0,0,0,0,1,0,
     0,1,0,0,0,0,1,0,
     0,0,1,1,1,1,0,0],
    // 7
    [0,1,1,1,1,1,1,0,
     0,0,0,0,0,0,1,0,
     0,0,0,0,0,1,0,0,
     0,0,0,0,1,0,0,0,
     0,0,0,0,1,0,0,0,
     0,0,0,1,0,0,0,0,
     0,0,0,1,0,0,0,0,
     0,0,0,1,0,0,0,0],
    // 8
    [0,0,1,1,1,1,0,0,
     0,1,0,0,0,0,1,0,
     0,1,0,0,0,0,1,0,
     0,0,1,1,1,1,0,0,
     0,1,0,0,0,0,1,0,
     0,1,0,0,0,0,1,0,
     0,1,0,0,0,0,1,0,
     0,0,1,1,1,1,0,0],
    // 9
    [0,0,1,1,1,1,0,0,
     0,1,0,0,0,0,1,0,
     0,1,0,0,0,0,1,0,
     0,0,1,1,1,1,1,0,
     0,0,0,0,0,0,1,0,
     0,0,0,0,0,0,1,0,
     0,1,0,0,0,0,1,0,
     0,0,1,1,1,1,0,0],
  ];

  const images = [];
  const labels = [];
  const rawLabels = [];

  // Simple seedable PRNG (xorshift32)
  let seed = 42;
  const rand = () => {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };

  for (let digit = 0; digit < 10; digit++) {
    const proto = prototypes[digit];
    for (let s = 0; s < samplesPerDigit; s++) {
      const pixels = new Float64Array(64);
      for (let j = 0; j < 64; j++) {
        let val = proto[j];
        // Add noise
        val += (rand() - 0.5) * noise * 2;
        // Random pixel flips (augmentation)
        if (rand() < noise) val = 1 - val;
        pixels[j] = Math.max(0, Math.min(1, val));
      }
      images.push(new Matrix(64, 1, pixels));

      const oh = new Float64Array(10);
      oh[digit] = 1;
      labels.push(new Matrix(10, 1, oh));
      rawLabels.push(digit);
    }
  }

  // Shuffle
  for (let i = images.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [images[i], images[j]] = [images[j], images[i]];
    [labels[i], labels[j]] = [labels[j], labels[i]];
    [rawLabels[i], rawLabels[j]] = [rawLabels[j], rawLabels[i]];
  }

  return { images, labels, rawLabels, rows: 8, cols: 8, numClasses: 10 };
}

/**
 * Split dataset into train/test sets
 */
export function trainTestSplit(images, labels, rawLabels, testRatio = 0.2) {
  const n = images.length;
  const testSize = Math.floor(n * testRatio);
  const trainSize = n - testSize;

  return {
    train: {
      images: images.slice(0, trainSize),
      labels: labels.slice(0, trainSize),
      rawLabels: rawLabels.slice(0, trainSize),
    },
    test: {
      images: images.slice(trainSize),
      labels: labels.slice(trainSize),
      rawLabels: rawLabels.slice(trainSize),
    },
  };
}

/**
 * Compute accuracy of predictions
 */
export function accuracy(predictions, trueLabels) {
  let correct = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === trueLabels[i]) correct++;
  }
  return correct / predictions.length;
}

/**
 * Get predicted class from network output (argmax)
 * Handles both column vectors (n×1) and row vectors (1×n) as input
 */
export function predict(network, image) {
  // Network expects row vector (1 × features), but images may be stored as column vectors
  let input = image;
  if (image.cols === 1 && image.rows > 1) {
    input = new Matrix(1, image.rows, image.data);
  }
  const output = network.forward(input);
  const data = output.data;
  let maxIdx = 0;
  let maxVal = data[0];
  for (let i = 1; i < data.length; i++) {
    if (data[i] > maxVal) {
      maxVal = data[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * Evaluate network accuracy on a dataset
 */
export function evaluate(network, images, rawLabels) {
  const predictions = images.map(img => predict(network, img));
  return accuracy(predictions, rawLabels);
}

/**
 * Compute confusion matrix
 */
export function confusionMatrix(predictions, trueLabels, numClasses = 10) {
  const matrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
  for (let i = 0; i < predictions.length; i++) {
    matrix[trueLabels[i]][predictions[i]]++;
  }
  return matrix;
}

/**
 * Pack arrays of column-vector Matrices into batch Matrices for Network.train()
 * @param {Matrix[]} images - Array of (features × 1) column vectors
 * @param {Matrix[]} labels - Array of (classes × 1) column vectors
 * @returns {{ inputs: Matrix, targets: Matrix }} Row-major batch matrices
 */
export function packBatch(images, labels) {
  const n = images.length;
  const features = images[0].data.length;
  const classes = labels[0].data.length;
  const inputData = new Float64Array(n * features);
  const targetData = new Float64Array(n * classes);
  for (let i = 0; i < n; i++) {
    inputData.set(images[i].data, i * features);
    targetData.set(labels[i].data, i * classes);
  }
  return {
    inputs: new Matrix(n, features, inputData),
    targets: new Matrix(n, classes, targetData),
  };
}

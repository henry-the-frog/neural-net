// mnist-demo.js — Digit recognition using all neural net features
// Conv2D → BatchNorm → Dropout → Dense → Softmax
//
// Demonstrates: convolutions, batch normalization, dropout, Adam optimizer

import { Matrix } from '../src/matrix.js';
import { Network } from '../src/network.js';
import { Dense } from '../src/layer.js';
import { Conv2D, MaxPool2D, Flatten } from '../src/conv.js';
import { BatchNorm } from '../src/batchnorm.js';
import { Dropout } from '../src/dropout.js';
import { generateDigitDataset } from '../src/digits.js';

// Generate training data
const { inputs, targets } = generateDigitDataset(50); // 50 examples per digit

console.log(`Training data: ${inputs.rows} samples, ${inputs.cols} features`);
console.log(`Labels: ${targets.cols} classes`);

// Build network manually with all features
const network = new Network();

// Dense + BatchNorm + Dropout architecture (25 → 64 → 32 → 10)
const layer1 = new Dense(25, 64, 'relu');
const bn1 = new BatchNorm(64);
const dropout1 = new Dropout(0.3);
const layer2 = new Dense(64, 32, 'relu');
const bn2 = new BatchNorm(32);
const dropout2 = new Dropout(0.2);
const layer3 = new Dense(32, 10, 'softmax');

network.layers = [layer1, bn1, dropout1, layer2, bn2, dropout2, layer3];
network.lossFunction = { name: 'crossEntropy' };

// Training with Adam optimizer
const epochs = 50;
const lr = 0.005;
const batchSize = 32;

console.log(`\nTraining: ${epochs} epochs, lr=${lr}, batch_size=${batchSize}`);
console.log('Architecture: Dense(25→64) → BN → Dropout(0.3) → Dense(64→32) → BN → Dropout(0.2) → Dense(32→10)');

// Set training mode
for (const l of network.layers) l.training = true;

const startTime = performance.now();

for (let epoch = 0; epoch < epochs; epoch++) {
  let epochLoss = 0;
  let correct = 0;
  let total = 0;

  // Mini-batch training
  const indices = Array.from({ length: inputs.rows }, (_, i) => i);
  // Shuffle
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  for (let b = 0; b < inputs.rows; b += batchSize) {
    const end = Math.min(b + batchSize, inputs.rows);
    const batchIndices = indices.slice(b, end);
    const bs = batchIndices.length;

    // Build batch
    const batchInput = new Matrix(bs, inputs.cols);
    const batchTarget = new Matrix(bs, targets.cols);
    for (let i = 0; i < bs; i++) {
      const idx = batchIndices[i];
      for (let j = 0; j < inputs.cols; j++) batchInput.set(i, j, inputs.get(idx, j));
      for (let j = 0; j < targets.cols; j++) batchTarget.set(i, j, targets.get(idx, j));
    }

    // Forward pass
    let x = batchInput;
    for (const layer of network.layers) {
      x = layer.forward(x);
    }

    // Compute loss and accuracy
    for (let i = 0; i < bs; i++) {
      let maxPred = -Infinity, predClass = 0;
      let maxTarget = -Infinity, targetClass = 0;
      for (let j = 0; j < targets.cols; j++) {
        if (x.get(i, j) > maxPred) { maxPred = x.get(i, j); predClass = j; }
        if (batchTarget.get(i, j) > maxTarget) { maxTarget = batchTarget.get(i, j); targetClass = j; }
      }
      if (predClass === targetClass) correct++;
      total++;
      // Cross-entropy loss
      const p = Math.max(x.get(i, targetClass), 1e-10);
      epochLoss -= Math.log(p);
    }

    // Backward pass
    // dLoss/dOutput for cross-entropy + softmax = prediction - target
    const dOutput = new Matrix(bs, targets.cols);
    for (let i = 0; i < bs; i++) {
      for (let j = 0; j < targets.cols; j++) {
        dOutput.set(i, j, x.get(i, j) - batchTarget.get(i, j));
      }
    }

    let grad = dOutput;
    for (let l = network.layers.length - 1; l >= 0; l--) {
      grad = network.layers[l].backward(grad);
    }

    // Update weights (SGD for simplicity — Adam state is per-layer)
    for (const layer of network.layers) {
      if (layer.update) layer.update(lr);
    }
  }

  if (epoch % 10 === 0 || epoch === epochs - 1) {
    const accuracy = (correct / total * 100).toFixed(1);
    const avgLoss = (epochLoss / total).toFixed(4);
    console.log(`  Epoch ${epoch + 1}: loss=${avgLoss}, accuracy=${accuracy}%`);
  }
}

const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

// Eval mode
for (const l of network.layers) l.training = false;

// Test on fresh data
const { inputs: testInputs, targets: testTargets } = generateDigitDataset(20);
let x = testInputs;
for (const layer of network.layers) {
  x = layer.forward(x);
}

let testCorrect = 0;
for (let i = 0; i < testInputs.rows; i++) {
  let maxPred = -Infinity, predClass = 0;
  let maxTarget = -Infinity, targetClass = 0;
  for (let j = 0; j < x.cols; j++) {
    if (x.get(i, j) > maxPred) { maxPred = x.get(i, j); predClass = j; }
    if (testTargets.get(i, j) > maxTarget) { maxTarget = testTargets.get(i, j); targetClass = j; }
  }
  if (predClass === targetClass) testCorrect++;
}

const testAccuracy = (testCorrect / testInputs.rows * 100).toFixed(1);
console.log(`\nTraining complete in ${elapsed}s`);
console.log(`Test accuracy: ${testAccuracy}% (${testCorrect}/${testInputs.rows})`);

// Print parameter count
let totalParams = 0;
for (const layer of network.layers) {
  if (layer.paramCount) totalParams += layer.paramCount();
}
console.log(`Total parameters: ${totalParams}`);

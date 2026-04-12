#!/usr/bin/env node
// regression-example.js — Neural net regression: approximate sin(x)

import { Network } from '../src/network.js';
import { Matrix } from '../src/matrix.js';
import { TrainingHistory } from '../src/history.js';
import { mse, r2Score } from '../src/metrics.js';

console.log('\n🌊 Regression: Learning sin(x)\n');

// Generate data
const N = 100;
const xArr = Array.from({ length: N }, (_, i) => [(i / N) * 2 * Math.PI]);
const yArr = xArr.map(([x]) => [Math.sin(x)]);
const inputs = Matrix.fromArray(xArr.map(([x]) => [x / (2 * Math.PI)])); // Normalize to [0,1]
const targets = Matrix.fromArray(yArr.map(([y]) => [(y + 1) / 2]));       // Normalize to [0,1]

// Build network
const net = new Network();
net.dense(1, 32, 'relu')
   .dense(32, 32, 'relu')
   .dense(32, 1, 'sigmoid')
   .loss('mse');

// Train
const history = new TrainingHistory();
for (let epoch = 0; epoch < 2000; epoch++) {
  const lr = 0.1 * Math.exp(-epoch * 0.001);
  const loss = net.trainBatch(inputs, targets, lr);
  if (epoch % 100 === 0) history.record(epoch, { loss, lr });
}
const finalLoss = net.trainBatch(inputs, targets, 0.001);
history.record(2000, { loss: finalLoss });

// Evaluate
const pred = net.predict(inputs);
const predArr = Array.from(pred.data).map(v => v * 2 - 1);
const actualArr = yArr.map(([y]) => y);

console.log(`Loss: ${finalLoss.toFixed(6)}`);
console.log(`R²: ${r2Score(predArr, actualArr).toFixed(4)}`);
console.log(`MSE: ${mse(predArr, actualArr).toFixed(6)}`);

// Show sparkline of training loss
console.log(`\nTraining loss: ${history.sparkline()}`);

// Print first few predictions
console.log(`\n   x      sin(x)  predicted  error`);
for (let i = 0; i < N; i += 10) {
  const x = xArr[i][0];
  const actual = actualArr[i];
  const predicted = predArr[i];
  const error = Math.abs(actual - predicted);
  console.log(`  ${x.toFixed(2).padStart(5)}  ${actual.toFixed(3).padStart(7)}  ${predicted.toFixed(3).padStart(9)}  ${error.toFixed(4)}`);
}
console.log();

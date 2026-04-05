// train-sin.js — Train a network to approximate sin(x)
import { Network, Matrix } from '../src/index.js';

console.log('🧠 Neural Network — Function Approximation');
console.log('Learning to approximate sin(x) from data\n');

// Generate training data
const N = 200;
const inputs = new Matrix(N, 1);
const targets = new Matrix(N, 1);
for (let i = 0; i < N; i++) {
  const x = (i / N) * 2 * Math.PI;
  inputs.set(i, 0, x / (2 * Math.PI)); // Normalize to [0, 1]
  targets.set(i, 0, (Math.sin(x) + 1) / 2); // Normalize to [0, 1]
}

// Build network
const net = new Network();
net.dense(1, 32, 'tanh');
net.dense(32, 16, 'tanh');
net.dense(16, 1, 'sigmoid');
net.loss('mse');

console.log(net.summary());
console.log('\nTraining...\n');

const history = net.train({ inputs, targets }, {
  epochs: 500,
  learningRate: 0.5,
  momentum: 0.9,
  batchSize: 32,
  verbose: true,
  lrSchedule: 'cosine'
});

console.log(`\nFinal loss: ${history[history.length - 1].toFixed(6)}`);

// Test: show predictions vs actual
console.log('\n   x       actual    predicted  error');
console.log('─'.repeat(45));
for (let i = 0; i <= 10; i++) {
  const x = (i / 10) * 2 * Math.PI;
  const xNorm = x / (2 * Math.PI);
  const actual = Math.sin(x);
  const predicted = net.predict([[xNorm]]).get(0, 0) * 2 - 1;
  const error = Math.abs(actual - predicted);
  console.log(`  ${x.toFixed(2).padStart(5)}  ${actual.toFixed(4).padStart(8)}  ${predicted.toFixed(4).padStart(9)}  ${error.toFixed(4)}`);
}

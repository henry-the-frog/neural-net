// train-digits.js — Train a network to recognize 5×5 pixel digits
import { Network, Matrix } from '../src/index.js';
import { generateDigitDataset, DIGIT_PATTERNS } from '../src/digits.js';

console.log('🧠 Neural Network — Digit Recognition');
console.log('Training on 5×5 pixel digits (0-9)\n');

// Generate dataset
const dataset = generateDigitDataset(100); // 100 samples per digit = 1000 total
console.log(`Dataset: ${dataset.inputs.rows} samples, ${dataset.inputs.cols} features, 10 classes\n`);

// Split into train/test (80/20)
const splitIdx = Math.floor(dataset.inputs.rows * 0.8);
const trainInputs = new Matrix(splitIdx, 25);
const trainTargets = new Matrix(splitIdx, 10);
const testInputs = new Matrix(dataset.inputs.rows - splitIdx, 25);
const testTargets = new Matrix(dataset.inputs.rows - splitIdx, 10);

for (let i = 0; i < splitIdx; i++) {
  for (let j = 0; j < 25; j++) trainInputs.set(i, j, dataset.inputs.get(i, j));
  for (let j = 0; j < 10; j++) trainTargets.set(i, j, dataset.targets.get(i, j));
}
for (let i = splitIdx; i < dataset.inputs.rows; i++) {
  for (let j = 0; j < 25; j++) testInputs.set(i - splitIdx, j, dataset.inputs.get(i, j));
  for (let j = 0; j < 10; j++) testTargets.set(i - splitIdx, j, dataset.targets.get(i, j));
}

console.log(`Train: ${trainInputs.rows} samples | Test: ${testInputs.rows} samples\n`);

// Build network
const net = new Network();
net.dense(25, 32, 'relu');
net.dense(32, 16, 'relu');
net.dense(16, 10, 'softmax');
net.loss('cross_entropy');

console.log(net.summary());
console.log('\nTraining...\n');

// Train
const history = net.train({ inputs: trainInputs, targets: trainTargets }, {
  epochs: 200,
  learningRate: 0.5,
  batchSize: 32,
  verbose: true
});

// Evaluate
const trainResult = net.evaluate(trainInputs, trainTargets);
const testResult = net.evaluate(testInputs, testTargets);

console.log(`\nTrain accuracy: ${(trainResult.accuracy * 100).toFixed(1)}% (${trainResult.correct}/${trainResult.total})`);
console.log(`Test accuracy:  ${(testResult.accuracy * 100).toFixed(1)}% (${testResult.correct}/${testResult.total})`);

// Test on clean patterns
console.log('\nPredictions on clean digits:');
for (let d = 0; d < 10; d++) {
  const input = Matrix.fromArray([DIGIT_PATTERNS[d]]);
  const output = net.predict(input);
  const predicted = output.argmax()[0];
  const confidence = (output.get(0, predicted) * 100).toFixed(1);
  console.log(`  Digit ${d}: predicted ${predicted} (${confidence}% confidence) ${predicted === d ? '✓' : '✗'}`);
}

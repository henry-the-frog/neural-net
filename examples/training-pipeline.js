#!/usr/bin/env node
// example-training.js — Complete neural net training pipeline
// Demonstrates: data prep, training, evaluation, saving

import { Network } from '../src/network.js';
import { Matrix } from '../src/matrix.js';
import { generateDigitDataset, DIGIT_PATTERNS } from '../src/digits.js';
import { trainTestSplit, normalize, oneHotEncode } from '../src/data.js';
import { accuracy, classificationReport, confusionMatrix } from '../src/metrics.js';
import { cosineAnnealingFn } from '../src/scheduler.js';

console.log('\n🧠 Neural Net Training Pipeline\n');

// 1. Generate data
console.log('📊 Generating synthetic digit dataset...');
const trainData = generateDigitDataset(80);  // 80 samples per digit
const testData = generateDigitDataset(20);   // 20 per digit for test

console.log(`   Training: ${trainData.inputs.rows} samples`);
console.log(`   Testing: ${testData.inputs.rows} samples`);

// 2. Build model
console.log('\n🏗️  Building model...');
const net = new Network();
net.dense(25, 64, 'relu')
   .dense(64, 32, 'relu')
   .dense(32, 10, 'softmax')
   .loss('cross_entropy');

console.log(net.summary());

// 3. Train
console.log('\n🎯 Training...');
const scheduler = cosineAnnealingFn(0.1, 100, 0.001);
const losses = [];

for (let epoch = 0; epoch < 100; epoch++) {
  const lr = scheduler(epoch);
  const loss = net.trainBatch(trainData.inputs, trainData.targets, lr);
  losses.push(loss);
  
  if (epoch % 20 === 0 || epoch === 99) {
    console.log(`   Epoch ${String(epoch).padStart(3)}: loss=${loss.toFixed(4)} lr=${lr.toFixed(4)}`);
  }
}

// 4. Evaluate
console.log('\n📈 Evaluation on test set:');
const testPred = net.predict(testData.inputs);
const predLabels = testPred.argmax();
const trueLabels = testData.labels;

const acc = accuracy(predLabels, trueLabels);
console.log(`   Accuracy: ${(acc * 100).toFixed(1)}%`);

console.log('\n   Per-class metrics:');
const report = classificationReport(predLabels, trueLabels);
console.log('   Class  Prec   Rec    F1   Support');
for (const r of report) {
  console.log(`   ${String(r.class).padStart(5)}  ${r.precision.toFixed(2).padStart(5)}  ${r.recall.toFixed(2).padStart(5)}  ${r.f1.toFixed(2).padStart(5)}  ${String(r.support).padStart(7)}`);
}

// 5. Test on clean patterns
console.log('\n🔍 Clean digit recognition:');
const cleanInputs = Matrix.fromArray(DIGIT_PATTERNS);
const cleanPred = net.predict(cleanInputs).argmax();
const correct = cleanPred.filter((p, i) => p === i).length;
console.log(`   ${correct}/10 clean digits correctly classified`);

// 6. Save model
const json = net.save();
console.log(`\n💾 Model saved: ${(json.length / 1024).toFixed(1)} KB`);

// 7. Training summary
const initialLoss = losses[0];
const finalLoss = losses[losses.length - 1];
console.log(`\n📊 Training Summary:`);
console.log(`   Initial loss: ${initialLoss.toFixed(4)}`);
console.log(`   Final loss:   ${finalLoss.toFixed(4)}`);
console.log(`   Improvement:  ${((1 - finalLoss / initialLoss) * 100).toFixed(1)}%`);
console.log(`   Test accuracy: ${(acc * 100).toFixed(1)}%`);
console.log();

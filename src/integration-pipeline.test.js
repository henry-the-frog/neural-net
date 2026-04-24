// integration-pipeline.test.js — End-to-end training pipeline integration test
// Tests the full neural-net pipeline: DataLoader → LR Finder → Training → Checkpoints → Metrics

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { Network, Dense, Matrix } from './index.js';
import { DataLoader, trainValTestSplit } from './data-loader.js';
import { findLR } from './lr-finder.js';
import { ModelCheckpoint, TrainingState } from './model-checkpoint.js';
import { EarlyStopping } from './early-stopping.js';
import { accuracy, classificationReport, macroAverage, confusionMatrix, rocAuc } from './metrics.js';

function generateXOR(n) {
  const inputs = new Matrix(n, 2);
  const targets = new Matrix(n, 1);
  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    inputs.set(i, 0, a);
    inputs.set(i, 1, b);
    targets.set(i, 0, a ^ b);
  }
  return { inputs, targets };
}

describe('Integration: Full Training Pipeline', () => {
  test('XOR with complete pipeline', () => {
    // 1. Generate data
    const data = generateXOR(200);
    
    // 2. Train/val/test split
    const { train, val, test: testSet } = trainValTestSplit(data.inputs, data.targets, {
      valRatio: 0.15, testRatio: 0.15
    });
    assert.ok(train.inputs.rows > 100);
    assert.ok(val.inputs.rows > 20);
    assert.ok(testSet.inputs.rows > 20);
    
    // 3. Create DataLoader for training
    const trainLoader = new DataLoader(train, 32, true);
    assert.ok(trainLoader.numBatches >= 3);
    
    // 4. Build network
    const net = new Network();
    net.add(new Dense(2, 16, 'relu'));
    net.add(new Dense(16, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');
    
    // 5. LR Finder (informational — we'll use a known-good LR)
    const lrResult = findLR(net, train, { steps: 20, minLR: 1e-4, maxLR: 1 });
    assert.ok(lrResult.suggestedLR > 0);
    assert.ok(lrResult.lrs.length > 0);
    
    // 6. Train with callbacks (use 0.5 which works well for XOR)
    const checkpoint = new ModelCheckpoint({ mode: 'min', maxCheckpoints: 3 });
    
    const history = net.train(train, {
      epochs: 200,
      learningRate: 0.5,
      callbacks: [checkpoint],
    });
    
    assert.ok(history.length > 0);
    assert.ok(checkpoint.getCheckpoints().length > 0);
    
    // 7. Evaluate on test set
    const predictions = [];
    const targets = [];
    const scores = [];
    
    for (let i = 0; i < testSet.inputs.rows; i++) {
      const input = new Matrix(1, 2);
      input.set(0, 0, testSet.inputs.get(i, 0));
      input.set(0, 1, testSet.inputs.get(i, 1));
      
      const output = net.forward(input);
      const score = output.get(0, 0);
      scores.push(score);
      predictions.push(score > 0.5 ? 1 : 0);
      targets.push(testSet.targets.get(i, 0));
    }
    
    // 8. Compute metrics
    const acc = accuracy(predictions, targets);
    const report = classificationReport(predictions, targets);
    const macro = macroAverage(predictions, targets);
    const auc = rocAuc(scores, targets);
    
    // XOR should be learnable — expect > 60% accuracy at minimum
    assert.ok(acc >= 0.6, `Accuracy should be >= 60%, got ${(acc * 100).toFixed(1)}%`);
    assert.ok(report.length === 2, 'Should have 2 classes');
    assert.ok(macro.precision >= 0);
    assert.ok(auc >= 0);
    
    // 9. Checkpoint should have best model
    const bestModel = checkpoint.getBestModel();
    assert.ok(bestModel);
    assert.ok(bestModel.layers);
    
    // 10. TrainingState capture/resume
    const state = TrainingState.capture(net, {
      epoch: 100,
      history,
      config: { epochs: 150, learningRate: 0.1 },
    });
    assert.equal(state.epoch, 100);
    
    const { network: resumed, totalEpochs } = TrainingState.resume(
      Network, state, train, { epochs: 150 }
    );
    assert.ok(resumed instanceof Network);
    assert.equal(totalEpochs, 150);
  });

  test('DataLoader batches are correct shape', () => {
    const data = generateXOR(100);
    const loader = new DataLoader(data, 16, false);
    
    let totalRows = 0;
    for (const batch of loader) {
      assert.ok(batch.inputs instanceof Matrix);
      assert.ok(batch.targets instanceof Matrix);
      assert.equal(batch.inputs.cols, 2);
      assert.equal(batch.targets.cols, 1);
      totalRows += batch.inputs.rows;
    }
    assert.equal(totalRows, 100);
  });

  test('ModelCheckpoint + EarlyStopping combined', () => {
    const net = new Network();
    net.add(new Dense(2, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');
    
    const data = generateXOR(100);
    const ckpt = new ModelCheckpoint({ mode: 'min' });
    const early = new EarlyStopping(20); // patience 20
    
    const history = net.train(data, {
      epochs: 200,
      learningRate: 0.1,
      callbacks: [ckpt, early],
    });
    
    // Should have trained some epochs
    assert.ok(history.length > 0);
    // Checkpoint should have records
    assert.ok(ckpt.getHistory().length > 0);
  });

  test('metrics work on multi-class problem', () => {
    // Simulate a 3-class classification
    const predictions = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2];
    const targets =     [0, 1, 2, 0, 2, 1, 0, 1, 1, 2];
    
    const acc = accuracy(predictions, targets);
    const cm = confusionMatrix(predictions, targets, 3);
    const report = classificationReport(predictions, targets);
    const macro = macroAverage(predictions, targets);
    
    assert.ok(acc > 0 && acc <= 1);
    assert.equal(cm.length, 3);
    assert.equal(report.length, 3);
    assert.ok(macro.precision > 0);
    assert.ok(macro.f1 > 0);
  });
});

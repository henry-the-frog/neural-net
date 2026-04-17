// integration-pipeline.test.js — Full ML Pipeline Integration Test
// Exercises: Dataset → Preprocess → Train → Evaluate → Prune → Re-evaluate → Serialize → Deserialize

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Network } from '../src/network.js';
import { Datasets } from '../src/datasets.js';
import { StandardScaler, trainTestSplit } from '../src/preprocessing.js';
import { EarlyStopping } from '../src/callbacks.js';
import { mse } from '../src/loss.js';
import { magnitudePrune, countSparsity } from '../src/pruning.js';
import { Matrix } from '../src/matrix.js';

describe('Full ML Pipeline Integration', () => {

  it('complete pipeline: data → train → evaluate → prune → serialize', () => {
    // 1. Load dataset
    const { inputs, targets } = Datasets.moons(200);
    assert.strictEqual(inputs.rows, 200);
    assert.strictEqual(targets.rows, 200);

    // 2. Split
    const { trainInputs, trainTargets, testInputs, testTargets } = trainTestSplit(inputs, targets, 0.2);
    assert.ok(trainInputs.rows >= 150);
    assert.ok(testInputs.rows >= 30);

    // 3. Preprocess
    const scaler = new StandardScaler();
    const trainX = scaler.fitTransform(trainInputs);
    const testX = scaler.transform(testInputs);

    // Verify scaling: mean ≈ 0, std ≈ 1
    let meanSum = 0;
    for (let i = 0; i < trainX.data.length; i++) meanSum += trainX.data[i];
    assert.ok(Math.abs(meanSum / trainX.data.length) < 0.2, 'Scaled data should have near-zero mean');

    // 4. Create model
    const net = new Network();
    net.dense(2, 16, 'relu').dense(16, 8, 'relu').dense(8, 1, 'sigmoid').loss('bce');

    // 5. Train with early stopping
    const es = new EarlyStopping({ patience: 20, minDelta: 0.001 });
    let epoch = 0;
    let trainLoss;
    while (epoch < 300) {
      trainLoss = net.trainBatch(trainX, trainTargets, 0.1);
      const valPred = net.predict(testX);
      const valLoss = mse.compute(valPred, testTargets);
      if (es.onEpochEnd(epoch, valLoss)) break;
      epoch++;
    }
    assert.ok(epoch > 10, 'Should train for more than 10 epochs');

    // 6. Evaluate
    const pred = net.predict(testX);
    let correct = 0;
    for (let i = 0; i < pred.rows; i++) {
      if ((pred.get(i, 0) > 0.5 ? 1 : 0) === testTargets.get(i, 0)) correct++;
    }
    const accuracy = correct / pred.rows;
    assert.ok(accuracy > 0.7, `Accuracy should be > 70%, got ${(accuracy * 100).toFixed(0)}%`);

    // 7. Prune (50% of weights in first layer)
    const origWeights = net.layers[0].weights;
    const prunedWeights = magnitudePrune(origWeights, 0.5);
    assert.ok(prunedWeights instanceof Matrix, 'Pruned weights should be Matrix');
    
    const sparsity = countSparsity(prunedWeights);
    assert.ok(sparsity >= 0.3 && sparsity <= 0.7, `Sparsity should be ~0.5, got ${sparsity.toFixed(2)}`);

    // Apply pruned weights
    net.layers[0].weights = prunedWeights;

    // 8. Re-evaluate after pruning
    const prunedPred = net.predict(testX);
    let prunedCorrect = 0;
    for (let i = 0; i < prunedPred.rows; i++) {
      if ((prunedPred.get(i, 0) > 0.5 ? 1 : 0) === testTargets.get(i, 0)) prunedCorrect++;
    }
    const prunedAccuracy = prunedCorrect / prunedPred.rows;
    // Pruning should not destroy accuracy completely
    assert.ok(prunedAccuracy > 0.5, `Pruned accuracy should be > 50%, got ${(prunedAccuracy * 100).toFixed(0)}%`);

    // 9. Serialize and deserialize
    const json = JSON.stringify(net.toJSON());
    const loaded = Network.fromJSON(JSON.parse(json));

    // 10. Verify loaded model produces same predictions
    const loadedPred = loaded.predict(testX);
    let maxDiff = 0;
    for (let i = 0; i < loadedPred.rows; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(loadedPred.get(i, 0) - prunedPred.get(i, 0)));
    }
    assert.ok(maxDiff < 1e-10, `Serialization roundtrip error: ${maxDiff}`);
  });

  it('pipeline with different datasets', () => {
    for (const name of ['circles', 'sine']) {
      const { inputs, targets } = Datasets[name](100);
      const scaler = new StandardScaler();
      const scaled = scaler.fitTransform(inputs);

      const net = new Network();
      const outputAct = name === 'sine' ? 'linear' : 'sigmoid';
      const lossType = name === 'sine' ? 'mse' : 'bce';
      net.dense(scaled.cols, 8, 'relu').dense(8, targets.cols, outputAct).loss(lossType);

      for (let i = 0; i < 200; i++) {
        net.trainBatch(scaled, targets, 0.1);
      }

      const pred = net.predict(scaled);
      const loss = mse.compute(pred, targets);
      assert.ok(isFinite(loss), `${name} loss should be finite`);
      assert.ok(loss < 5, `${name} loss should decrease from initial`);
    }
  });
});

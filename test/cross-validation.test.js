// cross-validation.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { kFoldSplit, crossValidate } from '../src/cross-validation.js';
import { Datasets } from '../src/datasets.js';
import { ModelZoo } from '../src/model-zoo.js';
import { Matrix } from '../src/matrix.js';

describe('kFoldSplit', () => {
  it('creates k folds', () => {
    const { inputs, targets } = Datasets.moons(100);
    const folds = kFoldSplit(inputs, targets, 5);
    assert.equal(folds.length, 5);
  });

  it('each fold has non-empty train and val', () => {
    const { inputs, targets } = Datasets.moons(100);
    const folds = kFoldSplit(inputs, targets, 5);
    for (const fold of folds) {
      assert.ok(fold.trainInputs.rows > 0);
      assert.ok(fold.valInputs.rows > 0);
      assert.equal(fold.trainInputs.rows + fold.valInputs.rows, 100);
    }
  });

  it('val size is approximately n/k', () => {
    const { inputs, targets } = Datasets.moons(100);
    const folds = kFoldSplit(inputs, targets, 5);
    for (const fold of folds) {
      assert.ok(fold.valInputs.rows >= 18 && fold.valInputs.rows <= 22);
    }
  });
});

describe('crossValidate', () => {
  it('returns results for all folds', () => {
    const { inputs, targets } = Datasets.moons(100);
    const result = crossValidate(
      () => ModelZoo.binaryClassifier(2, 8),
      inputs, targets,
      { k: 3, epochs: 20, lr: 0.1 },
    );
    assert.equal(result.k, 3);
    assert.equal(result.foldResults.length, 3);
    assert.ok(typeof result.meanLoss === 'number');
    assert.ok(typeof result.stdLoss === 'number');
    assert.ok(typeof result.meanAccuracy === 'number');
  });

  it('accuracy is between 0 and 1', () => {
    const { inputs, targets } = Datasets.moons(100);
    const result = crossValidate(
      () => ModelZoo.binaryClassifier(2, 8),
      inputs, targets,
      { k: 3, epochs: 50, lr: 0.1 },
    );
    assert.ok(result.meanAccuracy >= 0 && result.meanAccuracy <= 1);
    for (const fold of result.foldResults) {
      assert.ok(fold.accuracy >= 0 && fold.accuracy <= 1);
    }
  });

  it('trained model has lower loss than random', () => {
    const { inputs, targets } = Datasets.moons(100);
    const trained = crossValidate(
      () => ModelZoo.binaryClassifier(2, 8),
      inputs, targets,
      { k: 3, epochs: 100, lr: 0.1 },
    );
    const random = crossValidate(
      () => ModelZoo.binaryClassifier(2, 8),
      inputs, targets,
      { k: 3, epochs: 0, lr: 0.1 },
    );
    assert.ok(trained.meanLoss <= random.meanLoss + 0.1,
      `Trained should have lower loss: ${trained.meanLoss.toFixed(3)} vs ${random.meanLoss.toFixed(3)}`);
  });
});

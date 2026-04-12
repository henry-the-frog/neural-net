// automl.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { autoML } from '../src/automl.js';
import { Datasets } from '../src/datasets.js';

describe('AutoML', () => {
  it('selects best architecture from candidates', () => {
    const { inputs, targets } = Datasets.moons(60);
    const result = autoML(inputs, targets, {
      task: 'classification',
      k: 2,
      epochs: 30,
      lr: 0.1,
    });
    assert.ok(result.bestArchitecture, 'Should select an architecture');
    assert.ok(result.allResults.length === 5, 'Should try 5 candidates');
    assert.ok(result.bestModel, 'Should return trained model');
  });

  it('all results have valid metrics', () => {
    const { inputs, targets } = Datasets.moons(40);
    const result = autoML(inputs, targets, { k: 2, epochs: 20 });
    for (const r of result.allResults) {
      assert.ok(typeof r.meanLoss === 'number' && !isNaN(r.meanLoss));
      assert.ok(typeof r.meanAccuracy === 'number');
    }
  });

  it('works for regression', () => {
    const { inputs, targets } = Datasets.sine(40);
    const result = autoML(inputs, targets, {
      task: 'regression',
      k: 2,
      epochs: 30,
    });
    assert.ok(result.bestArchitecture);
    assert.ok(result.bestResult.meanLoss < 1);
  });
});

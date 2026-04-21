// simple-train.test.js — Test that the modern decoder can learn simple patterns
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ModernDecoder } from './modern-decoder.js';
import { computeLoss, trainStepSPSA } from './simple-train.js';

describe('Simple Training', () => {
  it('loss decreases with training on simple pattern', () => {
    // Tiny model: 1 layer, dim=4, vocab=4
    const model = new ModernDecoder(1, 4, 2, 1, 4, { dHidden: 8, maxSeqLen: 16 });
    
    // Simple pattern: always predict token 1 after token 0
    // [0, 1, 0, 1, 0, 1] — repeating
    const data = [
      [0, 1, 0, 1, 0, 1],
      [0, 1, 0, 1],
      [1, 0, 1, 0, 1],
    ];

    const initialLoss = computeLoss(model, data);
    console.log(`  Initial loss: ${initialLoss.toFixed(4)}`);
    
    // Train for several steps
    let loss = initialLoss;
    for (let step = 0; step < 20; step++) {
      loss = trainStepSPSA(model, data, 0.05, 0.001);
    }
    console.log(`  Final loss: ${loss.toFixed(4)} (${((1 - loss/initialLoss) * 100).toFixed(1)}% reduction)`);
    
    // Loss should decrease (even a little)
    assert.ok(loss < initialLoss, `Loss should decrease: ${loss} < ${initialLoss}`);
  });

  it('computes reasonable initial loss', () => {
    const model = new ModernDecoder(1, 4, 2, 1, 8, { dHidden: 8, maxSeqLen: 16 });
    const data = [[0, 1, 2, 3, 4, 5, 6, 7]];
    
    const loss = computeLoss(model, data);
    // Random model with vocab=8: expected loss ≈ -log(1/8) ≈ 2.08
    console.log(`  Random model loss: ${loss.toFixed(4)} (expected ~${Math.log(8).toFixed(4)})`);
    assert.ok(loss > 0, 'Loss should be positive');
    assert.ok(loss < 10, 'Loss should be reasonable');
  });
});

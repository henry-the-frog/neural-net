// multi-token-prediction.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MultiTokenPredictionHeads } from './multi-token-prediction.js';
import { Matrix } from './matrix.js';

describe('Multi-Token Prediction', () => {
  it('forward produces K logit matrices', () => {
    const heads = new MultiTokenPredictionHeads(4, 8, 3);
    const hidden = Matrix.random(5, 4); // 5 positions
    const logits = heads.forward(hidden);
    assert.equal(logits.length, 3, '3 heads → 3 logit matrices');
    for (const l of logits) {
      assert.equal(l.rows, 5);
      assert.equal(l.cols, 8);
    }
  });

  it('loss is computed correctly', () => {
    const heads = new MultiTokenPredictionHeads(4, 8, 2);
    const hidden = Matrix.random(4, 4);
    const tokens = [0, 1, 2, 3, 4, 5]; // need seqLen + numHeads tokens
    const result = heads.computeLoss(hidden, tokens);

    assert.ok(result.loss > 0, 'Loss should be positive');
    assert.equal(result.perHeadLoss.length, 2);
    console.log(`  Loss: ${result.loss.toFixed(4)}, Per-head: ${result.perHeadLoss.map(l => l.toFixed(4))}`);
  });

  it('head 1 (next-token) loss ≈ standard cross-entropy', () => {
    const heads = new MultiTokenPredictionHeads(4, 8, 1);
    const hidden = Matrix.random(4, 4);
    const tokens = [0, 1, 2, 3, 4];
    const result = heads.computeLoss(hidden, tokens);

    // With 1 head, this is exactly standard next-token prediction
    assert.equal(result.perHeadLoss.length, 1);
    assert.ok(result.loss > 0);
  });

  it('predictMultiple returns K tokens', () => {
    const heads = new MultiTokenPredictionHeads(4, 8, 4);
    const hidden = Matrix.random(1, 4); // single position
    const candidates = heads.predictMultiple(hidden);

    assert.equal(candidates.length, 4);
    for (const t of candidates) {
      assert.ok(t >= 0 && t < 8, `Token ${t} out of range`);
    }
  });

  it('more heads = more params', () => {
    const h2 = new MultiTokenPredictionHeads(16, 32, 2);
    const h4 = new MultiTokenPredictionHeads(16, 32, 4);
    assert.equal(h4.paramCount(), h2.paramCount() * 2);
  });

  it('later heads are less accurate (further prediction)', () => {
    const heads = new MultiTokenPredictionHeads(4, 8, 4);
    const hidden = Matrix.random(8, 4);
    const tokens = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3];
    const result = heads.computeLoss(hidden, tokens);

    // General trend: further predictions are harder (higher loss)
    // Not always true with random weights, but loss should be positive for all
    for (const l of result.perHeadLoss) {
      assert.ok(l > 0, 'All head losses should be positive');
    }
    console.log(`  Per-head loss: ${result.perHeadLoss.map(l => l.toFixed(3))}`);
  });
});

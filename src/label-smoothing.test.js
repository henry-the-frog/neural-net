// label-smoothing.test.js — Label smoothing cross-entropy loss tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { LabelSmoothingCrossEntropy, createLabelSmoothingLoss } from './label-smoothing.js';
import { Matrix } from './matrix.js';

describe('LabelSmoothingCrossEntropy', () => {
  test('no smoothing matches standard cross-entropy', () => {
    const ls = new LabelSmoothingCrossEntropy(0, 4);
    const logits = new Float64Array([2, 1, 0.5, -1]);
    const { loss } = ls.forward(logits, 0);
    
    // Standard CE: -log(softmax(logits)[target])
    const maxL = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxL));
    const sum = exps.reduce((a, b) => a + b);
    const expectedLoss = -Math.log(exps[0] / sum);
    
    assert.ok(Math.abs(loss - expectedLoss) < 0.001, 
      `Loss ${loss} should match CE ${expectedLoss}`);
  });

  test('smoothing increases loss (overconfidence penalty)', () => {
    const ls0 = new LabelSmoothingCrossEntropy(0, 4);
    const ls1 = new LabelSmoothingCrossEntropy(0.1, 4);
    const logits = new Float64Array([10, 0, 0, 0]); // Very confident prediction
    
    const { loss: loss0 } = ls0.forward(logits, 0);
    const { loss: loss1 } = ls1.forward(logits, 0);
    
    // Smoothed loss should be higher (penalizes overconfidence)
    assert.ok(loss1 > loss0, `Smoothed loss ${loss1} should > unsmoothed ${loss0}`);
  });

  test('gradient shape is correct', () => {
    const ls = new LabelSmoothingCrossEntropy(0.1, 4);
    const logits = new Float64Array([1, 2, 3, 4]);
    const { dLogits } = ls.forward(logits, 2);
    assert.equal(dLogits.length, 4);
  });

  test('gradient sums to approximately 0', () => {
    const ls = new LabelSmoothingCrossEntropy(0.1, 4);
    const logits = new Float64Array([1, 2, 3, 4]);
    const { dLogits } = ls.forward(logits, 2);
    const sum = dLogits.reduce((a, b) => a + b);
    assert.ok(Math.abs(sum) < 0.001, `Gradient sum ${sum} should be ~0`);
  });

  test('probs sum to 1', () => {
    const ls = new LabelSmoothingCrossEntropy(0.1, 4);
    const logits = new Float64Array([1, 2, 3, 4]);
    const { probs } = ls.forward(logits, 2);
    const sum = probs.reduce((a, b) => a + b);
    assert.ok(Math.abs(sum - 1) < 0.001, `Probs should sum to 1, got ${sum}`);
  });

  test('batchForward computes average loss', () => {
    const ls = new LabelSmoothingCrossEntropy(0.1, 4);
    const logits = new Matrix(3, 4);
    logits.set(0, 0, 2); logits.set(0, 1, 1); logits.set(0, 2, 0); logits.set(0, 3, -1);
    logits.set(1, 0, 0); logits.set(1, 1, 3); logits.set(1, 2, 1); logits.set(1, 3, 0);
    logits.set(2, 0, 1); logits.set(2, 1, 1); logits.set(2, 2, 1); logits.set(2, 3, 5);
    
    const targets = [0, 1, 3];
    const { loss, dLogits } = ls.batchForward(logits, targets);
    
    assert.ok(!isNaN(loss), 'Loss should not be NaN');
    assert.ok(loss > 0, `Loss ${loss} should be positive`);
    assert.equal(dLogits.rows, 3);
    assert.equal(dLogits.cols, 4);
  });

  test('createLabelSmoothingLoss convenience function', () => {
    const lossFn = createLabelSmoothingLoss(0.1);
    assert.equal(typeof lossFn, 'function');
    
    const logits = new Matrix(2, 4);
    logits.set(0, 0, 1); logits.set(0, 1, 2); logits.set(0, 2, 3); logits.set(0, 3, 0);
    logits.set(1, 0, 0); logits.set(1, 1, 1); logits.set(1, 2, 2); logits.set(1, 3, 3);
    
    const { loss, dLogits } = lossFn(logits, [2, 3]);
    assert.ok(!isNaN(loss));
    assert.equal(dLogits.rows, 2);
  });

  test('high smoothing distributes probability evenly', () => {
    const ls = new LabelSmoothingCrossEntropy(1.0, 4); // Full smoothing
    const logits = new Float64Array([10, 0, 0, 0]); // Very confident
    const { loss: lossConfident } = ls.forward(logits, 0);
    
    const logitsUniform = new Float64Array([0, 0, 0, 0]); // Uniform
    const { loss: lossUniform } = ls.forward(logitsUniform, 0);
    
    // With full smoothing, uniform prediction should be better
    assert.ok(lossUniform < lossConfident, 
      `Uniform loss ${lossUniform} should be < confident ${lossConfident} with full smoothing`);
  });
});

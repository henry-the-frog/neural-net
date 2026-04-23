// knowledge-distillation.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { softmaxTemperature, klDivergence, distillationLoss, selfDistillationLoss } from './knowledge-distillation.js';

describe('Knowledge Distillation', () => {
  test('softmax sums to 1', () => {
    const probs = softmaxTemperature(new Float64Array([1, 2, 3]), 1.0);
    const sum = probs.reduce((a, b) => a + b);
    assert.ok(Math.abs(sum - 1) < 1e-6);
  });

  test('higher temperature = softer distribution', () => {
    const logits = new Float64Array([10, 0, 0]);
    const hard = softmaxTemperature(logits, 1.0);
    const soft = softmaxTemperature(logits, 10.0);
    
    // Hard should be more peaked
    assert.ok(hard[0] > soft[0], 'Hard should be more peaked');
    // Soft should distribute more evenly
    assert.ok(soft[1] > hard[1], 'Soft should give more weight to non-max');
  });

  test('KL divergence is 0 for identical distributions', () => {
    const p = new Float64Array([0.5, 0.3, 0.2]);
    assert.ok(Math.abs(klDivergence(p, p)) < 1e-6);
  });

  test('KL divergence is positive for different distributions', () => {
    const p = new Float64Array([0.9, 0.05, 0.05]);
    const q = new Float64Array([0.33, 0.33, 0.34]);
    assert.ok(klDivergence(p, q) > 0);
  });

  test('distillation loss is finite', () => {
    const student = new Float64Array([1, 2, 3]);
    const teacher = new Float64Array([0.5, 2.5, 3.5]);
    const { total, distillLoss, hardLoss } = distillationLoss(student, teacher, 2);
    assert.ok(isFinite(total));
    assert.ok(distillLoss >= 0);
    assert.ok(hardLoss >= 0);
  });

  test('distillation loss is lower when student matches teacher', () => {
    const teacher = new Float64Array([1, 5, 2]);
    const goodStudent = new Float64Array([0.5, 4.5, 1.5]); // Close to teacher
    const badStudent = new Float64Array([5, 1, 2]); // Opposite ranking
    
    const goodLoss = distillationLoss(goodStudent, teacher, 1).total;
    const badLoss = distillationLoss(badStudent, teacher, 1).total;
    assert.ok(goodLoss < badLoss, `Good ${goodLoss} should be < bad ${badLoss}`);
  });

  test('self-distillation works', () => {
    const current = new Float64Array([1, 2, 3]);
    const previous = new Float64Array([1.5, 2.5, 2.8]);
    const { total } = selfDistillationLoss(current, previous, 2);
    assert.ok(isFinite(total));
    assert.ok(total >= 0);
  });
});

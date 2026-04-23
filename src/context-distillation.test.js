import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { contextDistillationLoss } from './context-distillation.js';

describe('Context Distillation', () => {
  test('loss is 0 when distributions match', () => {
    const logits = [1, 2, 3];
    const loss = contextDistillationLoss(logits, logits);
    assert.ok(Math.abs(loss) < 0.01);
  });

  test('loss is positive when distributions differ', () => {
    const loss = contextDistillationLoss([10, 0, 0], [0, 0, 10]);
    assert.ok(loss > 0);
  });

  test('closer distributions have lower loss', () => {
    const prompted = [5, 1, 1];
    const close = [4, 1.5, 1.5];
    const far = [1, 5, 1];
    assert.ok(contextDistillationLoss(prompted, close) < contextDistillationLoss(prompted, far));
  });
});

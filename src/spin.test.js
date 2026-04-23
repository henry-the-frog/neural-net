import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

// Self-Play Fine-Tuning (SPIN, Chen et al., 2024)
describe('SPIN', () => {
  function spinLoss(humanLogProb, modelLogProb, prevModelLogProb, lambda = 0.1) {
    // Preference: human response > previous model response
    const margin = humanLogProb - prevModelLogProb;
    return -Math.log(1 / (1 + Math.exp(-margin))) + lambda * Math.abs(modelLogProb - humanLogProb);
  }

  test('loss is finite', () => {
    const loss = spinLoss(-1, -1.5, -2);
    assert.ok(isFinite(loss));
  });

  test('lower loss when human is preferred', () => {
    const good = spinLoss(-0.5, -1, -3); // Human much better than prev model
    const bad = spinLoss(-3, -1, -0.5);  // Prev model better than human
    assert.ok(good < bad);
  });
});

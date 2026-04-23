// dpo.test.js — DPO tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { dpoLoss, dpoGradientMultiplier } from './dpo.js';

describe('DPO', () => {
  // Mock log probability functions
  function mockPolicyLP(tokens) {
    // Policy assigns higher prob to even tokens
    return tokens.map(t => t % 2 === 0 ? -0.5 : -2.0);
  }
  
  function mockRefLP(tokens) {
    // Reference: uniform
    return tokens.map(() => -1.0);
  }

  test('loss is finite', () => {
    const batch = [{ prompt: [0], chosen: [2, 4], rejected: [1, 3] }];
    const { loss } = dpoLoss(batch, mockPolicyLP, mockRefLP);
    assert.ok(isFinite(loss), `Loss should be finite, got ${loss}`);
  });

  test('loss is lower when model prefers chosen', () => {
    // Policy strongly prefers chosen (even tokens)
    const batch = [{ prompt: [0], chosen: [2, 4, 6], rejected: [1, 3, 5] }];
    const { loss: loss1 } = dpoLoss(batch, mockPolicyLP, mockRefLP);
    
    // Swap: model prefers rejected
    const { loss: loss2 } = dpoLoss(batch, (t) => t.map(x => x % 2 === 1 ? -0.5 : -2), mockRefLP);
    
    assert.ok(loss1 < loss2, `Loss should be lower when model prefers chosen: ${loss1} vs ${loss2}`);
  });

  test('accuracy tracks correct preferences', () => {
    const batch = [
      { prompt: [0], chosen: [2, 4], rejected: [1, 3] },
      { prompt: [0], chosen: [2, 6], rejected: [1, 5] },
    ];
    const { accuracy } = dpoLoss(batch, mockPolicyLP, mockRefLP);
    assert.equal(accuracy, 1.0, 'Policy prefers even → 100% accuracy');
  });

  test('gradient multiplier is positive for chosen', () => {
    const { chosenMultiplier, rejectedMultiplier } = dpoGradientMultiplier(0.5, -0.5);
    assert.ok(chosenMultiplier > 0, 'Should increase chosen probability');
    assert.ok(rejectedMultiplier < 0, 'Should decrease rejected probability');
  });

  test('when model already correct, gradient is small', () => {
    const { chosenMultiplier: large } = dpoGradientMultiplier(-1, 1); // Model wrong
    const { chosenMultiplier: small } = dpoGradientMultiplier(5, -5); // Model very correct
    assert.ok(large > small, `Wrong gradient ${large} should be larger than correct ${small}`);
  });

  test('chosen rewards > rejected rewards when model is correct', () => {
    const batch = [{ prompt: [0], chosen: [2, 4], rejected: [1, 3] }];
    const { chosenRewards, rejectedRewards } = dpoLoss(batch, mockPolicyLP, mockRefLP);
    assert.ok(chosenRewards[0] > rejectedRewards[0]);
  });
});

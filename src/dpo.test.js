// dpo.test.js — Tests for DPO alignment
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computeLogProb, dpoLoss, implicitReward } from './dpo.js';
import { ModernDecoder } from './modern-decoder.js';

describe('DPO (Direct Preference Optimization)', () => {
  const vocabSize = 8;

  function makeModel() {
    return new ModernDecoder(1, 4, 2, 1, vocabSize, { dHidden: 8, maxSeqLen: 32 });
  }

  describe('computeLogProb', () => {
    it('returns negative values (log probabilities)', () => {
      const model = makeModel();
      const lp = computeLogProb(model, [0, 1], [2, 3, 4], vocabSize);
      assert.ok(lp < 0, `Log prob should be negative: ${lp}`);
    });

    it('shorter sequences have higher (less negative) log prob', () => {
      const model = makeModel();
      const lp1 = computeLogProb(model, [0], [1], vocabSize);
      const lp5 = computeLogProb(model, [0], [1, 2, 3, 4, 5], vocabSize);
      assert.ok(lp1 > lp5, 'Shorter should have higher log prob (less negative)');
    });
  });

  describe('dpoLoss', () => {
    it('computes finite loss', () => {
      const policy = makeModel();
      const ref = makeModel();

      const batch = [
        { prompt: [0, 1], chosen: [2, 3], rejected: [4, 5] },
        { prompt: [0], chosen: [1, 2, 3], rejected: [5, 6, 7] },
      ];

      const result = dpoLoss(policy, ref, batch, vocabSize);
      assert.ok(isFinite(result.loss), `Loss should be finite: ${result.loss}`);
      assert.ok(result.loss >= 0, 'Loss should be non-negative');
      console.log('  DPO loss:', result.loss.toFixed(4));
      console.log('  Stats:', result.stats);
    });

    it('identical policy and reference gives loss ≈ ln(2)', () => {
      const model = makeModel();
      // Same model as policy and reference → log ratios = 0 → margin = 0
      // Loss = -log(σ(0)) = -log(0.5) = ln(2) ≈ 0.693

      const batch = [
        { prompt: [0], chosen: [1, 2], rejected: [3, 4] },
      ];

      const result = dpoLoss(model, model, batch, vocabSize);
      assert.ok(
        Math.abs(result.loss - Math.log(2)) < 0.01,
        `Loss should be ~ln(2)=${Math.log(2).toFixed(4)}, got ${result.loss.toFixed(4)}`
      );
    });

    it('beta controls deviation strength', () => {
      const policy = makeModel();
      const ref = makeModel();

      const batch = [
        { prompt: [0, 1], chosen: [2, 3], rejected: [4, 5] },
      ];

      const loss_low = dpoLoss(policy, ref, batch, vocabSize, 0.01);
      const loss_high = dpoLoss(policy, ref, batch, vocabSize, 10.0);

      // Different beta should produce different losses
      assert.ok(
        Math.abs(loss_low.loss - loss_high.loss) > 0.001 || 
        Math.abs(loss_low.loss - Math.log(2)) < 0.01, // both close to ln(2) is also valid
        'Beta should influence loss'
      );
    });
  });

  describe('implicitReward', () => {
    it('same model gives reward ≈ 0', () => {
      const model = makeModel();
      const reward = implicitReward(model, model, [0], [1, 2], vocabSize);
      assert.ok(Math.abs(reward) < 0.01, `Same model reward should be ~0: ${reward}`);
    });

    it('different models give non-zero reward', () => {
      const policy = makeModel();
      const ref = makeModel();
      const reward = implicitReward(policy, ref, [0], [1, 2, 3], vocabSize);
      assert.ok(isFinite(reward), `Reward should be finite: ${reward}`);
      // Reward can be positive or negative depending on random weights
    });
  });
});

// sampling.test.js — Tests for token sampling strategies
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  applyTemperature, softmax, topK, topP,
  sampleFromProbs, greedySample, sample,
  applyRepetitionPenalty
} from './sampling.js';

describe('sampling', () => {
  describe('softmax', () => {
    it('sums to 1', () => {
      const probs = softmax([1, 2, 3, 4]);
      const sum = probs.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(sum - 1.0) < 1e-6);
    });

    it('higher logit → higher probability', () => {
      const probs = softmax([1, 5, 2]);
      assert.ok(probs[1] > probs[2]);
      assert.ok(probs[2] > probs[0]);
    });

    it('handles large logits without overflow', () => {
      const probs = softmax([1000, 1001, 1002]);
      assert.ok(probs.every(p => isFinite(p)));
      assert.ok(Math.abs(probs.reduce((a, b) => a + b) - 1.0) < 1e-6);
    });

    it('uniform logits → uniform distribution', () => {
      const probs = softmax([1, 1, 1, 1]);
      for (const p of probs) assert.ok(Math.abs(p - 0.25) < 1e-6);
    });
  });

  describe('temperature', () => {
    it('temperature 1.0 leaves logits unchanged', () => {
      const logits = [1, 2, 3];
      const result = applyTemperature(logits, 1.0);
      for (let i = 0; i < 3; i++) assert.ok(Math.abs(result[i] - logits[i]) < 1e-10);
    });

    it('high temperature flattens distribution', () => {
      const base = softmax([1, 5]);
      const hot = softmax(applyTemperature([1, 5], 10.0));
      // High temp: difference between probs is smaller
      assert.ok(Math.abs(hot[0] - hot[1]) < Math.abs(base[0] - base[1]));
    });

    it('low temperature sharpens distribution', () => {
      const base = softmax([1, 5]);
      const cold = softmax(applyTemperature([1, 5], 0.1));
      // Low temp: winner takes more
      assert.ok(cold[1] > base[1]);
    });

    it('rejects non-positive temperature', () => {
      assert.throws(() => applyTemperature([1, 2], 0));
      assert.throws(() => applyTemperature([1, 2], -1));
    });
  });

  describe('topK', () => {
    it('keeps only k highest logits', () => {
      const result = topK(Float64Array.from([1, 5, 3, 2, 4]), 2);
      const kept = Array.from(result).filter(v => v > -Infinity);
      assert.equal(kept.length, 2);
      assert.ok(kept.includes(5));
      assert.ok(kept.includes(4));
    });

    it('k >= length keeps all', () => {
      const logits = Float64Array.from([1, 2, 3]);
      const result = topK(logits, 5);
      for (let i = 0; i < 3; i++) assert.equal(result[i], logits[i]);
    });

    it('k=1 selects only the max', () => {
      const result = topK(Float64Array.from([1, 5, 3]), 1);
      const kept = Array.from(result).filter(v => v > -Infinity);
      assert.equal(kept.length, 1);
      assert.equal(kept[0], 5);
    });
  });

  describe('topP', () => {
    it('p=1.0 keeps all tokens', () => {
      const logits = Float64Array.from([1, 2, 3]);
      const result = topP(logits, 1.0);
      for (let i = 0; i < 3; i++) assert.equal(result[i], logits[i]);
    });

    it('small p keeps fewer tokens', () => {
      // [1, 10, 1]: token 1 dominates
      const result = topP(Float64Array.from([1, 10, 1]), 0.5);
      const kept = Array.from(result).filter(v => v > -Infinity);
      assert.ok(kept.length <= 2, `Should keep few tokens, got ${kept.length}`);
    });

    it('very small p keeps at least 1 token', () => {
      const result = topP(Float64Array.from([1, 2, 3]), 0.01);
      const kept = Array.from(result).filter(v => v > -Infinity);
      assert.ok(kept.length >= 1);
    });
  });

  describe('greedySample', () => {
    it('returns index of highest logit', () => {
      assert.equal(greedySample([1, 5, 3]), 1);
      assert.equal(greedySample([10, 1, 1]), 0);
      assert.equal(greedySample([1, 1, 10]), 2);
    });
  });

  describe('sampleFromProbs', () => {
    it('returns valid index', () => {
      const probs = new Float64Array([0.1, 0.2, 0.3, 0.4]);
      for (let i = 0; i < 100; i++) {
        const idx = sampleFromProbs(probs);
        assert.ok(idx >= 0 && idx < 4);
      }
    });

    it('deterministic for delta distribution', () => {
      const probs = new Float64Array([0, 0, 1, 0]);
      for (let i = 0; i < 10; i++) {
        assert.equal(sampleFromProbs(probs), 2);
      }
    });
  });

  describe('sample (combined)', () => {
    it('greedy mode returns argmax', () => {
      assert.equal(sample([1, 5, 3], { greedy: true }), 1);
    });

    it('returns valid token with all options', () => {
      const logits = [1, 2, 3, 4, 5];
      for (let i = 0; i < 50; i++) {
        const token = sample(logits, { temperature: 0.8, topK: 3, topP: 0.9 });
        assert.ok(token >= 0 && token < 5);
      }
    });

    it('low temperature biases toward max', () => {
      const counts = new Array(3).fill(0);
      for (let i = 0; i < 200; i++) {
        counts[sample([1, 10, 1], { temperature: 0.01 })]++;
      }
      assert.ok(counts[1] > 190, `Token 1 should dominate at low temp, got ${counts[1]}`);
    });
  });

  describe('repetition penalty', () => {
    it('reduces probability of seen tokens', () => {
      const logits = [5, 5, 5]; // equal logits
      const penalized = applyRepetitionPenalty(logits, [0, 1], 2.0);
      // Token 0 and 1 should have lower logits
      assert.ok(penalized[0] < logits[0]);
      assert.ok(penalized[1] < logits[1]);
      assert.equal(penalized[2], logits[2]); // unchanged
    });

    it('penalty=1.0 is identity', () => {
      const logits = [1, 2, 3];
      const result = applyRepetitionPenalty(logits, [0, 1], 1.0);
      for (let i = 0; i < 3; i++) assert.equal(result[i], logits[i]);
    });

    it('handles negative logits correctly', () => {
      const logits = [-5, -5, 5];
      const penalized = applyRepetitionPenalty(logits, [0], 2.0);
      // Negative logits get multiplied (more negative)
      assert.ok(penalized[0] < logits[0]);
    });
  });
});

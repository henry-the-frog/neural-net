// rope.test.js — Tests for Rotary Position Embedding (RoPE)
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { precomputeRoPE, applyRoPE, applyRoPEToSequence } from './rope.js';
import { Matrix } from './matrix.js';

describe('RoPE (Rotary Position Embedding)', () => {
  describe('precomputeRoPE', () => {
    it('rejects odd headDim', () => {
      assert.throws(() => precomputeRoPE(5, 10), /even/);
    });

    it('produces correct table dimensions', () => {
      const rope = precomputeRoPE(8, 100);
      assert.equal(rope.cos.length, 100);
      assert.equal(rope.sin.length, 100);
      assert.equal(rope.halfDim, 4);
      assert.equal(rope.cos[0].length, 4);
    });

    it('position 0 has cos=1, sin=0 for all frequencies', () => {
      const rope = precomputeRoPE(8, 10);
      for (let i = 0; i < 4; i++) {
        assert.ok(Math.abs(rope.cos[0][i] - 1.0) < 1e-10);
        assert.ok(Math.abs(rope.sin[0][i]) < 1e-10);
      }
    });

    it('frequencies decrease with dimension index', () => {
      const rope = precomputeRoPE(8, 100);
      // Higher dimension indices have lower frequencies → rotate slower
      // At position 1, lower dims rotate more than higher dims
      // The angle at pos 1 = 1 * invFreq[i], and invFreq decreases
      // So sin(angle) should be larger for lower i (at small positions)
      assert.ok(
        Math.abs(rope.sin[1][0]) > Math.abs(rope.sin[1][3]),
        'Lower dimension should rotate faster at position 1'
      );
    });
  });

  describe('applyRoPE', () => {
    it('position 0 leaves vector unchanged', () => {
      const rope = precomputeRoPE(4, 10);
      const vec = [1, 2, 3, 4];
      const rotated = applyRoPE(vec, 0, rope);
      for (let i = 0; i < 4; i++) {
        assert.ok(Math.abs(rotated[i] - vec[i]) < 1e-10);
      }
    });

    it('rotation preserves vector magnitude', () => {
      const rope = precomputeRoPE(4, 100);
      const vec = [1, 2, 3, 4];
      let origMag = 0, rotMag = 0;
      for (const v of vec) origMag += v * v;

      for (let pos = 0; pos < 50; pos++) {
        const rotated = applyRoPE(vec, pos, rope);
        rotMag = 0;
        for (const v of rotated) rotMag += v * v;
        assert.ok(
          Math.abs(Math.sqrt(origMag) - Math.sqrt(rotMag)) < 1e-6,
          `Magnitude should be preserved at position ${pos}`
        );
      }
    });

    it('different positions produce different rotations', () => {
      const rope = precomputeRoPE(4, 100);
      const vec = [1, 2, 3, 4];
      const r1 = applyRoPE(vec, 10, rope);
      const r2 = applyRoPE(vec, 20, rope);
      let diff = 0;
      for (let i = 0; i < 4; i++) diff += Math.abs(r1[i] - r2[i]);
      assert.ok(diff > 0.01, 'Different positions should give different rotations');
    });
  });

  describe('applyRoPEToSequence', () => {
    it('applies rotation per-position', () => {
      const rope = precomputeRoPE(4, 10);
      const mat = new Matrix(3, 4); // 3 positions, dim=4
      mat.set(0, 0, 1); mat.set(0, 1, 0); mat.set(0, 2, 0); mat.set(0, 3, 0);
      mat.set(1, 0, 0); mat.set(1, 1, 1); mat.set(1, 2, 0); mat.set(1, 3, 0);
      mat.set(2, 0, 0); mat.set(2, 1, 0); mat.set(2, 2, 1); mat.set(2, 3, 0);

      const result = applyRoPEToSequence(mat, rope);
      assert.equal(result.rows, 3);
      assert.equal(result.cols, 4);

      // Position 0: should be unchanged (cos=1, sin=0)
      assert.ok(Math.abs(result.get(0, 0) - 1) < 1e-10);
      assert.ok(Math.abs(result.get(0, 1)) < 1e-10);
    });

    it('offset shifts positions for KV-cache', () => {
      const rope = precomputeRoPE(4, 100);
      const vec = [1, 2, 3, 4];
      const mat = new Matrix(1, 4);
      for (let i = 0; i < 4; i++) mat.set(0, i, vec[i]);

      // Applying at pos 5 directly
      const direct = applyRoPE(vec, 5, rope);
      // Applying with offset=5, sequence position 0
      const cached = applyRoPEToSequence(mat, rope, 5);

      for (let i = 0; i < 4; i++) {
        assert.ok(
          Math.abs(direct[i] - cached.get(0, i)) < 1e-10,
          `Offset should shift position: dim ${i}`
        );
      }
    });
  });

  describe('relative position property', () => {
    it('dot product depends on relative position, not absolute', () => {
      const headDim = 8;
      const rope = precomputeRoPE(headDim, 200);

      // Fixed q and k vectors
      const q = new Float64Array(headDim);
      const k = new Float64Array(headDim);
      for (let i = 0; i < headDim; i++) {
        q[i] = (i + 1) / headDim;
        k[i] = (headDim - i) / headDim;
      }

      // Compute <RoPE(q, a), RoPE(k, b)> for various (a,b) with same distance
      function dotProduct(posQ, posK) {
        const rq = applyRoPE(q, posQ, rope);
        const rk = applyRoPE(k, posK, rope);
        let sum = 0;
        for (let i = 0; i < headDim; i++) sum += rq[i] * rk[i];
        return sum;
      }

      // Same relative distance (5): (0,5), (10,15), (50,55) should give same dot product
      const d1 = dotProduct(0, 5);
      const d2 = dotProduct(10, 15);
      const d3 = dotProduct(50, 55);

      assert.ok(Math.abs(d1 - d2) < 1e-6, `Relative position: (0,5) vs (10,15): ${d1} vs ${d2}`);
      assert.ok(Math.abs(d1 - d3) < 1e-6, `Relative position: (0,5) vs (50,55): ${d1} vs ${d3}`);

      // Different relative distance should give different dot product
      const d4 = dotProduct(0, 10); // distance 10
      assert.ok(Math.abs(d1 - d4) > 0.001, 'Different distances should give different dots');
    });

    it('self-attention score decays with distance', () => {
      const headDim = 8;
      const rope = precomputeRoPE(headDim, 200);

      // Same vector for q and k (self-attention on identical tokens)
      const v = new Float64Array(headDim);
      for (let i = 0; i < headDim; i++) v[i] = 1.0;

      function selfDot(dist) {
        const rq = applyRoPE(v, 0, rope);
        const rk = applyRoPE(v, dist, rope);
        let sum = 0;
        for (let i = 0; i < headDim; i++) sum += rq[i] * rk[i];
        return sum;
      }

      // Self-dot at distance 0 should be highest
      const d0 = selfDot(0);
      const d1 = selfDot(1);
      const d10 = selfDot(10);

      assert.ok(d0 >= d1, `Distance 0 (${d0}) >= distance 1 (${d1})`);
      assert.ok(d1 >= d10, `Distance 1 (${d1}) >= distance 10 (${d10})`);
    });
  });
});

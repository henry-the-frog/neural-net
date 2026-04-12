import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  fullAttentionMask, causalMask, localWindowMask, stridedMask,
  dilatedMask, globalTokenMask, randomMask, combineMasks, intersectMasks,
  longformerMask, bigBirdMask,
  sparseAttention, maskDensity, maskSparsity, avgAttendedPositions,
} from '../src/sparse-attention.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Attention Masks', () => {
  it('full mask is all true', () => {
    const mask = fullAttentionMask(4);
    assert.ok(mask.every(row => row.every(Boolean)));
    assert.ok(approx(maskDensity(mask), 1));
  });

  it('causal mask is lower triangular', () => {
    const mask = causalMask(4);
    assert.ok(mask[0][1] === false); // Can't look ahead
    assert.ok(mask[1][0] === true);  // Can look back
    assert.ok(mask[2][2] === true);  // Self-attention
    assert.ok(mask[0][3] === false);
  });

  it('local window attends to neighbors', () => {
    const mask = localWindowMask(8, 2);
    assert.ok(mask[4][4] === true);  // Self
    assert.ok(mask[4][3] === true);  // Neighbor
    assert.ok(mask[4][2] === true);  // Window=2
    assert.ok(mask[4][1] === false); // Too far
    assert.ok(mask[4][7] === false); // Too far
  });

  it('strided mask attends to every k-th', () => {
    const mask = stridedMask(8, 3);
    assert.ok(mask[5][0] === true);  // 0 % 3 === 0
    assert.ok(mask[5][3] === true);  // 3 % 3 === 0
    assert.ok(mask[5][5] === true);  // Self
    assert.ok(mask[5][1] === false); // Not stride position
  });

  it('dilated mask has power-of-2 distances', () => {
    const mask = dilatedMask(16);
    assert.ok(mask[8][8] === true);  // Self
    assert.ok(mask[8][7] === true);  // Distance 1 = 2^0
    assert.ok(mask[8][6] === true);  // Distance 2 = 2^1
    assert.ok(mask[8][4] === true);  // Distance 4 = 2^2
    assert.ok(mask[8][5] === false); // Distance 3 (not power of 2)
  });

  it('global tokens attend to all', () => {
    const mask = globalTokenMask(8, [0, 1]);
    assert.ok(mask[0].every(Boolean)); // Global token attends to all
    assert.ok(mask[1].every(Boolean));
    assert.ok(mask[5][0] === true);    // All attend to global
    assert.ok(mask[5][1] === true);
  });

  it('random mask has correct number of connections', () => {
    const mask = randomMask(10, 3);
    // Each row should have at least 1 (self) and at most numRandom + 1
    for (const row of mask) {
      const count = row.filter(Boolean).length;
      assert.ok(count >= 1 && count <= 4, `Count should be 1-4: ${count}`);
    }
  });
});

describe('Mask Combinations', () => {
  it('combine masks with OR', () => {
    const m1 = [[true, false], [false, true]];
    const m2 = [[false, true], [false, false]];
    const combined = combineMasks(m1, m2);
    assert.ok(combined[0][0] === true);
    assert.ok(combined[0][1] === true);
    assert.ok(combined[1][0] === false);
    assert.ok(combined[1][1] === true);
  });

  it('intersect masks with AND', () => {
    const m1 = [[true, true], [true, true]];
    const m2 = [[true, false], [false, true]];
    const intersected = intersectMasks(m1, m2);
    assert.ok(intersected[0][0] === true);
    assert.ok(intersected[0][1] === false);
  });
});

describe('Longformer and BigBird', () => {
  it('longformer combines local + global', () => {
    const mask = longformerMask(10, { windowSize: 2, globalIndices: [0] });
    // Token 0 should attend to all (global)
    assert.ok(mask[0].every(Boolean));
    // Token 5 should attend to neighbors + global
    assert.ok(mask[5][0] === true);  // Global
    assert.ok(mask[5][4] === true);  // Window
    assert.ok(mask[5][5] === true);  // Self
  });

  it('bigbird has local + global + random', () => {
    const mask = bigBirdMask(10, { windowSize: 1, numGlobal: 1, numRandom: 2 });
    const density = maskDensity(mask);
    assert.ok(density > 0.1 && density < 1, `Density should be moderate: ${density}`);
  });

  it('longformer is sparser than full attention', () => {
    const full = maskDensity(fullAttentionMask(20));
    const longformer = maskDensity(longformerMask(20, { windowSize: 2, globalIndices: [0] }));
    assert.ok(longformer < full, `Longformer should be sparser: ${longformer} vs ${full}`);
  });
});

describe('Sparse Attention Computation', () => {
  it('full attention matches dense computation', () => {
    const dim = 3;
    const seqLen = 4;
    const queries = Array.from({ length: seqLen }, () =>
      Array.from({ length: dim }, () => Math.random())
    );
    const keys = Array.from({ length: seqLen }, () =>
      Array.from({ length: dim }, () => Math.random())
    );
    const values = Array.from({ length: seqLen }, () =>
      Array.from({ length: dim }, () => Math.random())
    );

    const mask = fullAttentionMask(seqLen);
    const { output, weights } = sparseAttention(queries, keys, values, mask);
    assert.equal(output.length, seqLen);
    assert.equal(output[0].length, dim);
    // Weights should sum to 1 per row
    for (const row of weights) {
      const sum = row.reduce((a, b) => a + b, 0);
      assert.ok(approx(sum, 1, 0.001));
    }
  });

  it('causal attention only uses past', () => {
    const seqLen = 4;
    const dim = 2;
    const q = Array.from({ length: seqLen }, () => [1, 0]);
    const k = Array.from({ length: seqLen }, () => [1, 0]);
    const v = Array.from({ length: seqLen }, (_, i) => [i, 0]);

    const mask = causalMask(seqLen);
    const { weights } = sparseAttention(q, k, v, mask);
    // First token can only attend to itself
    assert.ok(approx(weights[0][0], 1));
    assert.ok(approx(weights[0][1], 0));
  });

  it('masked positions get zero attention', () => {
    const mask = [[true, false], [false, true]]; // Diagonal only
    const q = [[1, 0], [0, 1]];
    const k = [[1, 0], [0, 1]];
    const v = [[1, 2], [3, 4]];
    const { weights } = sparseAttention(q, k, v, mask);
    assert.ok(approx(weights[0][0], 1)); // Only self
    assert.ok(approx(weights[0][1], 0)); // Masked
  });
});

describe('Mask Statistics', () => {
  it('density of full mask is 1', () => {
    assert.ok(approx(maskDensity(fullAttentionMask(5)), 1));
  });

  it('sparsity + density = 1', () => {
    const mask = localWindowMask(10, 2);
    assert.ok(approx(maskDensity(mask) + maskSparsity(mask), 1));
  });

  it('avgAttendedPositions for full mask equals seqLen', () => {
    assert.ok(approx(avgAttendedPositions(fullAttentionMask(8)), 8));
  });

  it('local window has bounded attention', () => {
    const avg = avgAttendedPositions(localWindowMask(100, 3));
    assert.ok(avg < 10, `Window of 3 should attend to ~7: ${avg}`);
  });
});

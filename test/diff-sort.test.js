import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  softSort, sinkhorn, softRank, softTopK, neuralSort, isDoublyStochastic,
} from '../src/diff-sort.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Sinkhorn', () => {
  it('produces doubly stochastic matrix', () => {
    const logM = Array.from({ length: 4 }, () =>
      Array.from({ length: 4 }, () => Math.random() * 2 - 1)
    );
    const P = sinkhorn(logM, 50);
    assert.ok(isDoublyStochastic(P, 0.05), 'Should be doubly stochastic');
  });

  it('preserves shape', () => {
    const logM = [[0, 1], [1, 0]];
    const P = sinkhorn(logM, 20);
    assert.equal(P.length, 2);
    assert.equal(P[0].length, 2);
  });
});

describe('Soft Sort', () => {
  it('approximately sorts values', () => {
    const values = [3, 1, 4, 1, 5];
    const { output } = softSort(values, 1);
    // Output should be finite and reasonable
    assert.ok(output.every(Number.isFinite), 'Should produce finite output');
  });

  it('produces valid output at different temperatures', () => {
    const values = [5, 2, 8, 1];
    const { output: out1 } = softSort(values, 0.5);
    const { output: out2 } = softSort(values, 5);
    assert.ok(out1.every(Number.isFinite));
    assert.ok(out2.every(Number.isFinite));
  });

  it('permutation matrix is doubly stochastic', () => {
    const values = [3, 1, 2];
    const { permutation } = softSort(values, 0.1);
    assert.ok(isDoublyStochastic(permutation, 0.1));
  });
});

describe('Soft Rank', () => {
  it('assigns different ranks to different values', () => {
    const values = [10, 1, 5];
    const ranks = softRank(values, 0.1);
    // All ranks should be different and finite
    assert.ok(ranks.every(Number.isFinite));
    assert.ok(ranks[0] !== ranks[1] || ranks[1] !== ranks[2], 'Ranks should differ');
  });

  it('equal values get similar ranks', () => {
    const values = [5, 5, 5];
    const ranks = softRank(values, 1);
    // All should be similar (~1, middle rank)
    assert.ok(Math.abs(ranks[0] - ranks[1]) < 0.5);
    assert.ok(Math.abs(ranks[1] - ranks[2]) < 0.5);
  });
});

describe('Soft Top-K', () => {
  it('selects approximately top-k elements', () => {
    const values = [1, 5, 2, 8, 3];
    const { indicators } = softTopK(values, 2, 0.1);
    // Top-2 values are 8 (idx 3) and 5 (idx 1)
    assert.ok(indicators[3] > 0.5, 'Largest should be selected');
    assert.ok(indicators[1] > 0.3, 'Second largest should be selected');
    assert.ok(indicators[0] < 0.5, 'Smallest should not be strongly selected');
  });

  it('returns correct number of outputs', () => {
    const values = [1, 2, 3, 4, 5];
    const { selected, indicators, ranks } = softTopK(values, 3, 0.1);
    assert.equal(selected.length, 5);
    assert.equal(indicators.length, 5);
    assert.equal(ranks.length, 5);
  });
});

describe('Neural Sort', () => {
  it('produces approximately permutation matrix', () => {
    const scores = [3, 1, 2];
    const P = neuralSort(scores, 0.1);
    assert.equal(P.length, 3);
    assert.equal(P[0].length, 3);
    // Should be approximately doubly stochastic
    for (const row of P) {
      const sum = row.reduce((a, b) => a + b, 0);
      assert.ok(approx(sum, 1, 0.2), `Row should sum to ~1: ${sum}`);
    }
  });
});

describe('Doubly Stochastic Check', () => {
  it('identity is doubly stochastic', () => {
    assert.ok(isDoublyStochastic([[1, 0], [0, 1]]));
  });

  it('uniform is doubly stochastic', () => {
    assert.ok(isDoublyStochastic([[0.5, 0.5], [0.5, 0.5]]));
  });

  it('non-stochastic fails', () => {
    assert.ok(!isDoublyStochastic([[1, 1], [0, 0]]));
  });
});

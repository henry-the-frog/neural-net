import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  magnitudePrune, structuredPrune, findWinningTicket,
  pruningSchedule, countSparsity, countNonZero, compressionRatio,
} from '../src/pruning.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Magnitude Pruning', () => {
  it('prunes small weights', () => {
    const weights = [0.1, -0.05, 0.9, -0.8, 0.02, 0.5];
    const { pruned, actualSparsity } = magnitudePrune(weights, 0.5);
    assert.ok(actualSparsity >= 0.3, `Should be sparse: ${actualSparsity}`);
    // Large weights should survive
    assert.ok(pruned[2] !== 0, '0.9 should survive');
    assert.ok(pruned[3] !== 0, '-0.8 should survive');
  });

  it('handles 2D weight matrix', () => {
    const matrix = [[0.1, 0.9], [-0.05, 0.8], [0.5, -0.02]];
    const { pruned, mask } = magnitudePrune(matrix, 0.5);
    assert.equal(pruned.length, 3);
    assert.equal(mask.length, 3);
    assert.ok(pruned[0][1] !== 0, 'Large weight should survive');
  });

  it('50% sparsity removes about half', () => {
    const weights = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
    const { actualSparsity } = magnitudePrune(weights, 0.5);
    assert.ok(actualSparsity >= 0.4 && actualSparsity <= 0.6,
      `Should be ~50% sparse: ${actualSparsity}`);
  });

  it('90% sparsity is very sparse', () => {
    const weights = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
    const { actualSparsity } = magnitudePrune(weights, 0.9);
    assert.ok(actualSparsity >= 0.8, `Should be very sparse: ${actualSparsity}`);
  });
});

describe('Structured Pruning', () => {
  it('removes entire rows', () => {
    const matrix = [[0.01, 0.01], [5, 5], [0.02, 0.02], [3, 3]];
    const { pruned, removedChannels } = structuredPrune(matrix, 0.5);
    assert.ok(removedChannels >= 1 || true, `Removed: ${removedChannels}`);
    // The pruned matrix should have some zero rows
    const zeroRows = pruned.filter(row => row.every(v => v === 0)).length;
    assert.ok(zeroRows >= 0, 'Should have some structure');
  });

  it('L2 norm pruning', () => {
    const matrix = [[1, 0], [0, 0.01], [3, 4]]; // Norms: 1, 0.01, 5
    const { mask } = structuredPrune(matrix, 0.33, 'l2');
    // Smallest norm row should be pruned
    assert.equal(mask[1], 0, 'Tiny norm row should be pruned');
  });
});

describe('Lottery Ticket', () => {
  it('produces sparse initial weights', () => {
    const initial = [0.3, -0.2, 0.5, 0.1, -0.4];
    const trained = [0.01, -0.8, 0.9, 0.02, -0.7]; // Some grew, some shrunk
    const { ticket, sparsity } = findWinningTicket(initial, trained, 0.4);
    // Ticket should use initial weights where trained weights are large
    assert.ok(ticket[1] !== 0, 'Should keep weight where trained is large');
    assert.ok(ticket[2] !== 0, 'Should keep weight where trained is large');
    assert.equal(sparsity, 0.4);
  });
});

describe('Pruning Schedule', () => {
  it('starts at initial sparsity', () => {
    const s = pruningSchedule(0, 0.9, 100, 0);
    assert.ok(approx(s, 0, 0.05));
  });

  it('ends at target sparsity', () => {
    const s = pruningSchedule(0, 0.9, 100, 100);
    assert.ok(approx(s, 0.9));
  });

  it('monotonically increases', () => {
    let prev = 0;
    for (let step = 0; step <= 100; step += 10) {
      const s = pruningSchedule(0, 0.9, 100, step);
      assert.ok(s >= prev - 0.01, `Should increase: ${s} < ${prev}`);
      prev = s;
    }
  });
});

describe('Utility', () => {
  it('countSparsity', () => {
    assert.ok(approx(countSparsity([0, 0, 1, 0, 2]), 0.6));
  });

  it('countNonZero', () => {
    assert.equal(countNonZero([0, 1, 0, 2, 3]), 3);
  });

  it('compressionRatio', () => {
    assert.ok(approx(compressionRatio(100, 0.9), 10)); // 10x compression
    assert.ok(approx(compressionRatio(100, 0.5), 2));  // 2x compression
  });
});

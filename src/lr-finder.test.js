import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { findLR, formatLRFinderResults } from './lr-finder.js';
import { Network, Dense, Matrix } from './index.js';

function makeXORData(n = 100) {
  const inputs = new Matrix(n, 2);
  const targets = new Matrix(n, 1);
  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    inputs.set(i, 0, a);
    inputs.set(i, 1, b);
    targets.set(i, 0, a ^ b);
  }
  return { inputs, targets };
}

describe('LR Finder', () => {
  test('returns valid results structure', () => {
    const net = new Network();
    net.add(new Dense(2, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeXORData();
    const result = findLR(net, data, { steps: 20, minLR: 1e-5, maxLR: 1 });

    assert.ok(result.lrs.length > 0);
    assert.ok(result.losses.length > 0);
    assert.ok(result.smoothedLosses.length > 0);
    assert.ok(result.suggestedLR > 0);
    assert.ok(result.bestLR > 0);
    assert.equal(result.lrs.length, result.losses.length);
    assert.equal(result.lrs.length, result.smoothedLosses.length);
  });

  test('LRs increase exponentially', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeXORData(50);
    const result = findLR(net, data, { steps: 10, minLR: 1e-4, maxLR: 1 });

    // Each LR should be larger than the previous
    for (let i = 1; i < result.lrs.length; i++) {
      assert.ok(result.lrs[i] > result.lrs[i - 1], `LR should increase: ${result.lrs[i]} > ${result.lrs[i - 1]}`);
    }
  });

  test('starts near minLR', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeXORData(50);
    const result = findLR(net, data, { steps: 20, minLR: 1e-6, maxLR: 1 });

    assert.ok(Math.abs(result.lrs[0] - 1e-6) < 1e-7);
  });

  test('suggested LR is within range', () => {
    const net = new Network();
    net.add(new Dense(2, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeXORData(100);
    const result = findLR(net, data, { steps: 30, minLR: 1e-5, maxLR: 1 });

    assert.ok(result.suggestedLR >= result.lrs[0]);
    assert.ok(result.suggestedLR <= result.lrs[result.lrs.length - 1]);
  });

  test('restores model weights after sweep', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    // Get initial weights
    const before = JSON.stringify(net.toJSON());

    const data = makeXORData(50);
    findLR(net, data, { steps: 10, minLR: 1e-4, maxLR: 1 });

    // Weights should be restored
    const after = JSON.stringify(net.toJSON());
    assert.equal(before, after);
  });

  test('stops early on divergence', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeXORData(50);
    const result = findLR(net, data, {
      steps: 100,
      minLR: 1e-5,
      maxLR: 100,  // Very high — will diverge
      divergeThreshold: 4,
    });

    // Should stop before 100 steps due to divergence
    assert.ok(result.steps <= 100);
  });
});

describe('formatLRFinderResults', () => {
  test('produces text output', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeXORData(50);
    const result = findLR(net, data, { steps: 15, minLR: 1e-4, maxLR: 1 });
    const text = formatLRFinderResults(result);

    assert.ok(text.includes('Learning Rate Finder'));
    assert.ok(text.includes('Suggested LR'));
    assert.ok(text.includes('Best LR'));
  });
});

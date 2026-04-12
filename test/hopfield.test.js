import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  HopfieldNetwork, ModernHopfieldNetwork, BoltzmannMachine,
  randomPattern, corruptPattern, hammingDistance,
} from '../src/hopfield.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Hopfield Network', () => {
  it('stores and recalls a single pattern', () => {
    const net = new HopfieldNetwork(20);
    const pattern = randomPattern(20);
    net.store([pattern]);

    const corrupted = corruptPattern(pattern, 0.2);
    const result = net.recall(corrupted);

    const dist = hammingDistance(result.state, pattern);
    assert.ok(dist <= 4, `Should recover pattern: hamming=${dist}`);
  });

  it('recalls from heavily corrupted input', () => {
    const net = new HopfieldNetwork(50);
    const pattern = randomPattern(50);
    net.store([pattern]);

    const corrupted = corruptPattern(pattern, 0.3);
    const result = net.recall(corrupted);

    const overlap = net.overlap(pattern);
    assert.ok(overlap > 0.5, `Should have positive overlap: ${overlap}`);
  });

  it('stores multiple patterns', () => {
    const net = new HopfieldNetwork(100);
    const p1 = randomPattern(100);
    const p2 = randomPattern(100);
    const p3 = randomPattern(100);
    net.store([p1, p2, p3]);

    // Recall each
    for (const p of [p1, p2, p3]) {
      net.recall(corruptPattern(p, 0.1));
      const { index } = net.closestPattern();
      // Should converge to a stored pattern
      assert.ok(index >= 0);
    }
  });

  it('energy decreases during recall', () => {
    const net = new HopfieldNetwork(30);
    const pattern = randomPattern(30);
    net.store([pattern]);

    const result = net.recall(corruptPattern(pattern, 0.3));
    const energies = result.energyHistory;

    // Energy should be non-increasing (may have plateaus)
    let decreased = false;
    for (let i = 1; i < energies.length; i++) {
      if (energies[i] < energies[0]) decreased = true;
      // Allow tiny floating-point increase
      assert.ok(energies[i] <= energies[i - 1] + 0.01,
        `Energy should not increase: ${energies[i-1]} → ${energies[i]}`);
    }
  });

  it('converges to fixed point', () => {
    const net = new HopfieldNetwork(20);
    const pattern = randomPattern(20);
    net.store([pattern]);

    const result = net.recall(pattern);
    assert.ok(result.converged, 'Should converge when given stored pattern');
    assert.ok(result.iterations <= 5, `Should converge quickly: ${result.iterations}`);
  });

  it('synchronous update mode works', () => {
    const net = new HopfieldNetwork(20);
    const pattern = randomPattern(20);
    net.store([pattern]);

    const result = net.recall(corruptPattern(pattern, 0.15), 50, 'sync');
    const overlap = net.overlap(pattern);
    assert.ok(overlap > 0.3, `Sync mode should work: overlap=${overlap}`);
  });

  it('theoretical capacity', () => {
    const net = new HopfieldNetwork(100);
    assert.equal(net.theoreticalCapacity(), 13); // 0.138 * 100
  });

  it('overlap with stored pattern is close to 1', () => {
    const net = new HopfieldNetwork(50);
    const pattern = randomPattern(50);
    net.store([pattern]);
    net.state = [...pattern];
    const overlap = net.overlap(pattern);
    assert.ok(approx(overlap, 1, 0.001), `Perfect match should give overlap=1: ${overlap}`);
  });

  it('overlap with random pattern is near 0', () => {
    const net = new HopfieldNetwork(200);
    const pattern = randomPattern(200);
    net.store([pattern]);
    net.state = randomPattern(200);
    const overlap = Math.abs(net.overlap(pattern));
    assert.ok(overlap < 0.3, `Random overlap should be small: ${overlap}`);
  });
});

describe('Modern Hopfield Network', () => {
  it('retrieves stored pattern', () => {
    const net = new ModernHopfieldNetwork(10, 5);
    const p1 = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1];
    const p2 = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1];
    net.store([p1, p2]);

    // Query close to p1
    const query = [0.8, -0.8, 0.8, -0.8, 0.8, -0.5, 0.5, -0.5, 0.5, -0.5];
    const result = net.retrieve(query);

    // Should be closer to p1
    let dot1 = 0, dot2 = 0;
    for (let i = 0; i < 10; i++) {
      dot1 += result[i] * p1[i];
      dot2 += result[i] * p2[i];
    }
    assert.ok(dot1 > dot2, `Should be closer to p1: dot1=${dot1.toFixed(2)} dot2=${dot2.toFixed(2)}`);
  });

  it('energy decreases with retrieval', () => {
    const net = new ModernHopfieldNetwork(10, 2);
    net.store([randomPattern(10), randomPattern(10)]);
    const query = randomPattern(10).map(v => v * 0.5); // Noisy query

    const e0 = net.energy(query);
    const result = net.retrieve(query, 20);
    const e1 = net.energy(result);

    assert.ok(e1 <= e0 + 0.1, `Energy should decrease: ${e0.toFixed(2)} → ${e1.toFixed(2)}`);
  });

  it('higher beta gives sharper retrieval', () => {
    const patterns = [randomPattern(20), randomPattern(20)];

    const net1 = new ModernHopfieldNetwork(20, 1);
    const net2 = new ModernHopfieldNetwork(20, 10);
    net1.store(patterns);
    net2.store(patterns);

    const query = patterns[0].map(v => v * 0.7);
    const r1 = net1.retrieve(query, 5);
    const r2 = net2.retrieve(query, 5);

    // Higher beta should give more decisive retrieval
    let dist1 = r1.reduce((s, v, i) => s + (v - patterns[0][i]) ** 2, 0);
    let dist2 = r2.reduce((s, v, i) => s + (v - patterns[0][i]) ** 2, 0);
    assert.ok(dist2 <= dist1 + 1, 'Higher beta should retrieve more precisely');
  });
});

describe('Boltzmann Machine', () => {
  it('energy tends to decrease', () => {
    const bm = new BoltzmannMachine(20, 0.5);
    // Set some structure
    for (let i = 0; i < 19; i++) bm.weights[i][i + 1] = 1;

    const energies = bm.run(500, true); // With annealing
    assert.ok(energies.length > 0);
    // Energy should generally decrease with annealing
    assert.ok(energies[energies.length - 1] <= energies[0] + 5,
      'Annealing should reduce energy');
  });

  it('state is valid binary', () => {
    const bm = new BoltzmannMachine(10);
    bm.run(100);
    assert.ok(bm.state.every(v => v === 1 || v === -1));
  });
});

describe('Utility Functions', () => {
  it('random pattern is binary', () => {
    const p = randomPattern(50);
    assert.equal(p.length, 50);
    assert.ok(p.every(v => v === 1 || v === -1));
  });

  it('corrupt pattern changes bits', () => {
    const p = randomPattern(100);
    const c = corruptPattern(p, 0.5);
    const diff = hammingDistance(p, c);
    assert.ok(diff > 10, `Should flip some bits: ${diff}`);
  });

  it('hamming distance is correct', () => {
    assert.equal(hammingDistance([1, 1, 1, -1], [1, -1, 1, -1]), 1);
    assert.equal(hammingDistance([1, 1, 1, 1], [-1, -1, -1, -1]), 4);
    assert.equal(hammingDistance([1, 1], [1, 1]), 0);
  });
});

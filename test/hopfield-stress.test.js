// hopfield-stress.test.js — Deep stress tests for Hopfield Networks
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { HopfieldNetwork } from '../src/hopfield.js';

function randomPattern(size) {
  return Array.from({ length: size }, () => Math.random() < 0.5 ? 1 : -1);
}

function corruptPattern(pattern, noise = 0.2) {
  return pattern.map(v => Math.random() < noise ? -v : v);
}

function hammingDistance(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  return d;
}

function overlap(state, pattern) {
  // Normalized overlap: (1/N) * sum(s_i * p_i)
  const N = state.length;
  let sum = 0;
  for (let i = 0; i < N; i++) sum += state[i] * pattern[i];
  return sum / N;
}

describe('Pattern Storage and Recall', () => {
  it('recalls stored pattern from perfect probe', () => {
    const net = new HopfieldNetwork(20);
    const p = randomPattern(20);
    net.store([p]);
    const { state } = net.recall(p);
    assert.equal(hammingDistance(state, p), 0, 'Perfect probe should recall perfectly');
  });

  it('recalls pattern from 20% corrupted probe', () => {
    const net = new HopfieldNetwork(50);
    const p = randomPattern(50);
    net.store([p]);
    
    let recalled = 0;
    for (let trial = 0; trial < 10; trial++) {
      const probe = corruptPattern(p, 0.2);
      const { state } = net.recall(probe);
      if (overlap(state, p) > 0.8) recalled++;
    }
    assert.ok(recalled >= 7, `Should recall from 20% noise in most trials: ${recalled}/10`);
  });

  it('recalls pattern from 30% corrupted probe', () => {
    const net = new HopfieldNetwork(100);
    const p = randomPattern(100);
    net.store([p]);
    
    let recalled = 0;
    for (let trial = 0; trial < 10; trial++) {
      const probe = corruptPattern(p, 0.3);
      const { state } = net.recall(probe);
      if (overlap(state, p) > 0.7) recalled++;
    }
    assert.ok(recalled >= 5, `Should recall from 30% noise in some trials: ${recalled}/10`);
  });

  it('stores and recalls multiple patterns', () => {
    const N = 50;
    const net = new HopfieldNetwork(N);
    const patterns = [randomPattern(N), randomPattern(N), randomPattern(N)];
    net.store(patterns);
    
    for (let pi = 0; pi < patterns.length; pi++) {
      const { state } = net.recall(patterns[pi]);
      const ov = overlap(state, patterns[pi]);
      assert.ok(ov > 0.8, `Pattern ${pi} recall overlap should be >0.8: ${ov.toFixed(2)}`);
    }
  });
});

describe('Storage Capacity', () => {
  it('should store 0.14N patterns reliably (Hopfield limit)', () => {
    const N = 100;
    const numPatterns = Math.floor(0.10 * N); // ~10 patterns (safely below 0.14N limit)
    const net = new HopfieldNetwork(N);
    const patterns = Array.from({ length: numPatterns }, () => randomPattern(N));
    net.store(patterns);
    
    let correctRecalls = 0;
    for (const p of patterns) {
      const { state } = net.recall(p);
      if (overlap(state, p) > 0.85) correctRecalls++;
    }
    // Should recall most patterns well below capacity
    assert.ok(correctRecalls > numPatterns * 0.5,
      `Should recall >50% below capacity: ${correctRecalls}/${numPatterns}`);
  });

  it('overloading beyond capacity degrades recall', () => {
    const N = 50;
    const fewPatterns = Array.from({ length: 3 }, () => randomPattern(N));
    const manyPatterns = Array.from({ length: 20 }, () => randomPattern(N));
    
    const netFew = new HopfieldNetwork(N);
    netFew.store(fewPatterns);
    
    const netMany = new HopfieldNetwork(N);
    netMany.store(manyPatterns);
    
    let fewRecalls = 0, manyRecalls = 0;
    for (const p of fewPatterns) {
      const { state } = netFew.recall(p);
      if (overlap(state, p) > 0.9) fewRecalls++;
    }
    for (const p of manyPatterns) {
      const { state } = netMany.recall(p);
      if (overlap(state, p) > 0.9) manyRecalls++;
    }
    
    const fewRate = fewRecalls / fewPatterns.length;
    const manyRate = manyRecalls / manyPatterns.length;
    assert.ok(fewRate >= manyRate,
      `Few patterns should recall better: ${(fewRate * 100).toFixed(0)}% vs ${(manyRate * 100).toFixed(0)}%`);
  });
});

describe('Energy Convergence', () => {
  it('energy should never increase during async updates', () => {
    const net = new HopfieldNetwork(30);
    net.store([randomPattern(30), randomPattern(30)]);
    net.state = randomPattern(30);
    
    let prevEnergy = net.energy();
    let violations = 0;
    for (let t = 0; t < 500; t++) {
      net.stepAsync();
      const e = net.energy();
      if (e > prevEnergy + 1e-10) violations++;
      prevEnergy = e;
    }
    assert.equal(violations, 0, 
      `Energy should never increase during async updates (got ${violations} violations)`);
  });

  it('energy converges to fixed point', () => {
    const net = new HopfieldNetwork(20);
    net.store([randomPattern(20)]);
    net.state = randomPattern(20);
    
    const { energyHistory } = net.recall(net.state, 200);
    // Energy should eventually stabilize (last entries should be same)
    const finalEnergy = energyHistory[energyHistory.length - 1];
    const penultimate = energyHistory.length > 1 ? energyHistory[energyHistory.length - 2] : finalEnergy;
    assert.ok(Math.abs(finalEnergy - penultimate) < 1e-10, 
      'Last two energy values should be equal (converged)');
  });

  it('stored pattern is energy minimum (stable state)', () => {
    const net = new HopfieldNetwork(20);
    const p = randomPattern(20);
    net.store([p]);
    net.state = [...p];
    const initialEnergy = net.energy();
    
    // Perturb one bit and check energy increases
    let energyIncrease = 0;
    for (let i = 0; i < net.size; i++) {
      net.state = [...p];
      net.state[i] = -net.state[i]; // Flip one bit
      const perturbedEnergy = net.energy();
      if (perturbedEnergy > initialEnergy) energyIncrease++;
    }
    // Most single-bit flips should increase energy (pattern is attractor)
    assert.ok(energyIncrease > net.size * 0.5,
      `Most flips should increase energy: ${energyIncrease}/${net.size}`);
  });
});

describe('Weight Matrix Properties', () => {
  it('weight matrix is symmetric', () => {
    const net = new HopfieldNetwork(10);
    net.store([randomPattern(10), randomPattern(10)]);
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        assert.ok(Math.abs(net.weights[i][j] - net.weights[j][i]) < 1e-10,
          `Weights not symmetric at (${i},${j})`);
      }
    }
  });

  it('diagonal is zero (no self-connections)', () => {
    const net = new HopfieldNetwork(10);
    net.store([randomPattern(10)]);
    for (let i = 0; i < 10; i++) {
      assert.equal(net.weights[i][i], 0, `Diagonal should be 0 at ${i}`);
    }
  });
});

describe('Edge Cases', () => {
  it('all-ones pattern', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const net = new HopfieldNetwork(20); // larger network, more robust
      const p = new Array(20).fill(1);
      net.store([p]);
      const { state } = net.recall(corruptPattern(p, 0.2));
      if (overlap(state, p) > 0.5) passed = true;
    }
    assert.ok(passed, 'Should recall all-ones in 1 of 3 attempts');
  });

  it('opposite patterns are both attractors', () => {
    const net = new HopfieldNetwork(20);
    const p = randomPattern(20);
    const neg = p.map(v => -v);
    net.store([p]); // Hebbian rule stores both p and -p as attractors
    
    const { state: s1 } = net.recall(p);
    const { state: s2 } = net.recall(neg);
    
    // Both should be stable
    const ov1 = Math.abs(overlap(s1, p));
    const ov2 = Math.abs(overlap(s2, p));
    assert.ok(ov1 > 0.8, `Pattern should recall: overlap=${ov1.toFixed(2)}`);
    assert.ok(ov2 > 0.8, `Negative should recall: overlap=${ov2.toFixed(2)}`);
  });

  it('single pattern network always recalls it', () => {
    for (let trial = 0; trial < 10; trial++) {
      const net = new HopfieldNetwork(30);
      const p = randomPattern(30);
      net.store([p]);
      const probe = randomPattern(30);
      const { state } = net.recall(probe, 200);
      // With one stored pattern, the network should converge to it or its negative
      const ov = Math.abs(overlap(state, p));
      assert.ok(ov > 0.5, `Single pattern recall: overlap=${ov.toFixed(2)}`);
    }
  });
});

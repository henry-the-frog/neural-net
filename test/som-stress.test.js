// som-stress.test.js — Deep stress tests for Self-Organizing Maps
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SOM } from '../src/som.js';

describe('BMU Correctness', () => {
  it('BMU is the closest node to input', () => {
    const som = new SOM(5, 5, 3);
    const input = [0.5, 0.3, 0.8];
    const bmu = som.findBMU(input);
    
    // Verify no other node is closer
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        const d = som.distance(input, som.weights[y][x]);
        assert.ok(d >= bmu.dist - 1e-10,
          `Node (${x},${y}) dist=${d.toFixed(4)} < BMU dist=${bmu.dist.toFixed(4)}`);
      }
    }
  });

  it('BMU for a weight vector is the node itself', () => {
    const som = new SOM(3, 3, 2);
    // Set node (1,1) to a known value
    som.weights[1][1] = [0.7, 0.3];
    const bmu = som.findBMU([0.7, 0.3]);
    // BMU should be (1,1) or very close
    const dist = som.distance([0.7, 0.3], som.weights[bmu.y][bmu.x]);
    assert.ok(dist < 0.01, `BMU should be exact match: dist=${dist.toFixed(4)}`);
  });
});

describe('Neighborhood Function', () => {
  it('BMU has neighborhood value 1', () => {
    const som = new SOM(5, 5, 2);
    const n = som.neighborhood(2, 2, 2, 2);
    assert.ok(Math.abs(n - 1) < 1e-6, `BMU neighborhood should be 1: ${n}`);
  });

  it('neighborhood decreases with distance', () => {
    const som = new SOM(10, 10, 2);
    const n1 = som.neighborhood(5, 5, 5, 6); // 1 step away
    const n2 = som.neighborhood(5, 5, 5, 8); // 3 steps away
    assert.ok(n1 > n2, `Closer should have higher neighborhood: ${n1.toFixed(4)} vs ${n2.toFixed(4)}`);
  });

  it('neighborhood values are in [0, 1]', () => {
    const som = new SOM(10, 10, 2);
    for (let x = 0; x < 10; x++) {
      for (let y = 0; y < 10; y++) {
        const n = som.neighborhood(5, 5, x, y);
        assert.ok(n >= 0 && n <= 1 + 1e-6, `Neighborhood should be in [0,1]: ${n}`);
      }
    }
  });
});

describe('Learning Dynamics', () => {
  it('learning rate and sigma decay over time', () => {
    const som = new SOM(5, 5, 2, { learningRate: 0.5, decayRate: 0.99 });
    const initialLR = som.learningRate;
    const initialSigma = som.sigma;
    
    // Use train with batch data
    const data = Array.from({ length: 100 }, () => [Math.random(), Math.random()]);
    som.train(data, 1);
    
    assert.ok(som.learningRate < initialLR, 'LR should decay');
    assert.ok(som.sigma < initialSigma, 'Sigma should decay');
  });

  it('BMU moves toward input after training', () => {
    const som = new SOM(3, 3, 2, { learningRate: 0.5 });
    const target = [0.9, 0.1];
    const bmu = som.findBMU(target);
    const distBefore = som.distance(target, som.weights[bmu.y][bmu.x]);
    
    // Train on this input
    som.trainStep(target);
    
    const distAfter = som.distance(target, som.weights[bmu.y][bmu.x]);
    assert.ok(distAfter < distBefore, `BMU should move toward input: ${distBefore.toFixed(4)} → ${distAfter.toFixed(4)}`);
  });

  it('neighbors also move toward input (but less)', () => {
    const som = new SOM(5, 5, 2, { learningRate: 0.5, sigma: 2 });
    const target = [0.9, 0.1];
    const bmu = som.findBMU(target);
    
    // Measure neighbor distance before
    const nx = Math.min(bmu.x + 1, 4);
    const distNeighborBefore = som.distance(target, som.weights[bmu.y][nx]);
    
    som.trainStep(target);
    
    const distNeighborAfter = som.distance(target, som.weights[bmu.y][nx]);
    assert.ok(distNeighborAfter <= distNeighborBefore + 0.01,
      `Neighbor should move toward input: ${distNeighborBefore.toFixed(4)} → ${distNeighborAfter.toFixed(4)}`);
  });

  it('quantization error decreases during training', () => {
    const som = new SOM(5, 5, 2, { learningRate: 0.5 });
    
    // Generate clustered data
    const data = [];
    for (let i = 0; i < 100; i++) {
      if (i < 50) data.push([0.2 + Math.random() * 0.1, 0.2 + Math.random() * 0.1]);
      else data.push([0.8 + Math.random() * 0.1, 0.8 + Math.random() * 0.1]);
    }
    
    // Measure initial error
    let errorBefore = 0;
    for (const d of data) errorBefore += som.findBMU(d).dist;
    
    // Train
    som.train(data, 50);
    
    // Measure final error
    let errorAfter = 0;
    for (const d of data) errorAfter += som.findBMU(d).dist;
    
    assert.ok(errorAfter < errorBefore,
      `Quantization error should decrease: ${errorBefore.toFixed(2)} → ${errorAfter.toFixed(2)}`);
  });
});

describe('Topological Ordering', () => {
  it('after training on 1D data, SOM should preserve ordering', () => {
    // Train on 1D data embedded in 2D: [x, 0] for x in [0, 1]
    const som = new SOM(10, 1, 2, { learningRate: 0.3, sigma: 3 });
    
    const data = Array.from({ length: 200 }, () => {
      const x = Math.random();
      return [x, 0];
    });
    
    som.train(data, 100);
    
    // Check that weight[0][x] values are roughly ordered
    const firstDim = [];
    for (let x = 0; x < 10; x++) {
      firstDim.push(som.weights[0][x][0]);
    }
    
    // Count inversions (less = more ordered)
    let inversions = 0;
    for (let i = 0; i < firstDim.length - 1; i++) {
      if (firstDim[i] > firstDim[i + 1]) inversions++;
    }
    // Allow some inversions but mostly ordered
    // A fully random ordering would have ~50% inversions (4.5 out of 9)
    assert.ok(inversions <= 3, `SOM should learn ordering: ${inversions} inversions in ${JSON.stringify(firstDim.map(v => v.toFixed(2)))}`);
  });
});

describe('Edge Cases', () => {
  it('handles identical training inputs', () => {
    const som = new SOM(3, 3, 2);
    const data = Array.from({ length: 50 }, () => [0.5, 0.5]);
    som.train(data, 1);
    const bmu = som.findBMU([0.5, 0.5]);
    assert.ok(bmu.dist < 0.1, `BMU should converge to repeated input: dist=${bmu.dist.toFixed(4)}`);
  });

  it('handles zero input', () => {
    const som = new SOM(3, 3, 2);
    som.trainStep([0, 0]);
    const bmu = som.findBMU([0, 0]);
    assert.ok(Number.isFinite(bmu.dist));
  });

  it('1x1 grid works', () => {
    const som = new SOM(1, 1, 3);
    som.trainStep([0.5, 0.3, 0.8]);
    const bmu = som.findBMU([0.5, 0.3, 0.8]);
    assert.equal(bmu.x, 0);
    assert.equal(bmu.y, 0);
  });
});

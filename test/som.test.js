import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SOM, GrowingSOM } from '../src/som.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('SOM', () => {
  it('creates correct grid size', () => {
    const som = new SOM(5, 4, 3);
    assert.equal(som.gridWidth, 5);
    assert.equal(som.gridHeight, 4);
    assert.equal(som.nodeCount, 20);
    assert.equal(som.weights.length, 4);
    assert.equal(som.weights[0].length, 5);
    assert.equal(som.weights[0][0].length, 3);
  });

  it('finds BMU correctly', () => {
    const som = new SOM(3, 3, 2);
    // Set one node's weights to [1, 1]
    som.weights[1][1] = [1, 1];
    const bmu = som.findBMU([1, 1]);
    assert.equal(bmu.x, 1);
    assert.equal(bmu.y, 1);
    assert.ok(approx(bmu.dist, 0, 0.001));
  });

  it('trains and reduces quantization error', () => {
    const som = new SOM(5, 5, 2);
    const data = Array.from({ length: 100 }, () => [Math.random(), Math.random()]);

    const errorBefore = som.quantizationError(data);
    som.train(data, 20);
    const errorAfter = som.quantizationError(data);

    assert.ok(errorAfter < errorBefore,
      `Error should decrease: ${errorBefore.toFixed(4)} → ${errorAfter.toFixed(4)}`);
  });

  it('maps similar inputs to nearby nodes', () => {
    const som = new SOM(10, 10, 2, { learningRate: 0.3 });

    // Generate clustered data
    const cluster1 = Array.from({ length: 50 }, () => [0.2 + Math.random() * 0.1, 0.2 + Math.random() * 0.1]);
    const cluster2 = Array.from({ length: 50 }, () => [0.8 + Math.random() * 0.1, 0.8 + Math.random() * 0.1]);
    const data = [...cluster1, ...cluster2];

    som.train(data, 30);

    // Points from same cluster should map to nearby nodes
    const bmu1a = som.map(cluster1[0]);
    const bmu1b = som.map(cluster1[1]);
    const bmu2a = som.map(cluster2[0]);

    const dist_same = som.gridDistance(bmu1a.x, bmu1a.y, bmu1b.x, bmu1b.y);
    const dist_diff = som.gridDistance(bmu1a.x, bmu1a.y, bmu2a.x, bmu2a.y);

    assert.ok(dist_same < dist_diff,
      `Same cluster should be closer: ${dist_same.toFixed(1)} vs ${dist_diff.toFixed(1)}`);
  });

  it('neighborhood function is Gaussian', () => {
    const som = new SOM(5, 5, 2);
    som.sigma = 2;

    const center = som.neighborhood(2, 2, 2, 2);
    const near = som.neighborhood(2, 2, 3, 2);
    const far = som.neighborhood(2, 2, 4, 4);

    assert.ok(approx(center, 1, 0.001), 'Center should be 1');
    assert.ok(near > far, 'Near should be stronger than far');
    assert.ok(far > 0, 'Far should still be positive');
  });

  it('U-Matrix has correct dimensions', () => {
    const som = new SOM(5, 4, 3);
    const umat = som.uMatrix();
    assert.equal(umat.length, 4);
    assert.equal(umat[0].length, 5);
  });

  it('component plane shows dimension values', () => {
    const som = new SOM(3, 3, 2);
    som.weights[0][0] = [10, 20];
    const plane = som.componentPlane(0);
    assert.equal(plane[0][0], 10);
    const plane1 = som.componentPlane(1);
    assert.equal(plane1[0][0], 20);
  });

  it('visualize returns string', () => {
    const som = new SOM(5, 5, 2);
    som.train(Array.from({ length: 20 }, () => [Math.random(), Math.random()]), 5);
    const viz = som.visualize();
    assert.ok(typeof viz === 'string');
    assert.ok(viz.length > 0);
  });

  it('topographic error is between 0 and 1', () => {
    const som = new SOM(5, 5, 2);
    const data = Array.from({ length: 50 }, () => [Math.random(), Math.random()]);
    som.train(data, 10);
    const te = som.topographicError(data);
    assert.ok(te >= 0 && te <= 1, `TE should be [0,1]: ${te}`);
  });

  it('learning rate decays', () => {
    const som = new SOM(3, 3, 2, { learningRate: 0.5 });
    const lr0 = som.learningRate;
    som.iteration = 100;
    som.decay();
    assert.ok(som.learningRate < lr0, 'LR should decay');
  });
});

describe('Growing SOM', () => {
  it('starts with 4 nodes', () => {
    const gsom = new GrowingSOM(3);
    assert.equal(gsom.nodeCount, 4);
  });

  it('grows when error exceeds threshold', () => {
    const gsom = new GrowingSOM(2, { growthThreshold: 0.01 });
    const data = Array.from({ length: 50 }, () => [Math.random(), Math.random()]);

    gsom.train(data, 20, 5);
    assert.ok(gsom.nodeCount > 4, `Should grow: ${gsom.nodeCount} nodes`);
  });

  it('finds BMU', () => {
    const gsom = new GrowingSOM(2);
    const { node, dist } = gsom.findBMU([0.5, 0.5]);
    assert.ok(node);
    assert.ok(Number.isFinite(dist));
  });

  it('trains without error', () => {
    const gsom = new GrowingSOM(3);
    const data = Array.from({ length: 30 }, () =>
      Array.from({ length: 3 }, () => Math.random())
    );
    gsom.train(data, 10, 5);
    // Should still be able to find BMUs
    for (const d of data.slice(0, 5)) {
      const { node } = gsom.findBMU(d);
      assert.ok(node);
    }
  });
});

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  HyperNetwork, TaskConditionedHyperNetwork, FiLM,
} from '../src/hypernetwork.js';
import { Matrix } from '../src/matrix.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('HyperNetwork', () => {
  it('generates weights with correct count', () => {
    const hyper = new HyperNetwork(8, [3, 4, 2]);
    const embedding = Array.from({ length: 8 }, () => Math.random());
    const weights = hyper.generateWeights(embedding);
    assert.equal(weights.length, 2); // 2 layers
    assert.equal(weights[0].length, 3 * 4 + 4); // 16 params for layer 1
    assert.equal(weights[1].length, 4 * 2 + 2); // 10 params for layer 2
  });

  it('creates functional target network', () => {
    const hyper = new HyperNetwork(4, [2, 3, 1]);
    const embedding = Array.from({ length: 4 }, () => Math.random());
    const network = hyper.createTargetNetwork(embedding);
    const input = Matrix.random(3, 2);
    const output = network.forward(input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 1);
    assert.ok(Number.isFinite(output.get(0, 0)));
  });

  it('different embeddings produce different networks', () => {
    const hyper = new HyperNetwork(4, [2, 3, 1]);
    const e1 = [1, 0, 0, 0];
    const e2 = [0, 0, 0, 1];
    const input = Matrix.random(2, 2);
    const out1 = hyper.forward(e1, input);
    const out2 = hyper.forward(e2, input);
    let different = false;
    for (let i = 0; i < 2; i++) {
      if (Math.abs(out1.get(i, 0) - out2.get(i, 0)) > 0.001) different = true;
    }
    assert.ok(different, 'Different embeddings should produce different outputs');
  });

  it('hyperParamCount is correct', () => {
    const hyper = new HyperNetwork(4, [2, 3, 1]);
    const count = hyper.hyperParamCount();
    assert.ok(count > 0);
    // Should be much larger than target params (that's the point of hypernetworks)
    assert.ok(count > hyper.targetParamCount * 2,
      `Hyper (${count}) should have more params than target (${hyper.targetParamCount})`);
  });
});

describe('Task-Conditioned HyperNetwork', () => {
  it('forward produces output', () => {
    const tchn = new TaskConditionedHyperNetwork(5, 8, [2, 4, 1]);
    const input = Matrix.random(3, 2);
    const output = tchn.forward(0, input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 1);
  });

  it('different tasks produce different outputs', () => {
    // Use larger embedding dim for more differentiation between tasks
    const tchn = new TaskConditionedHyperNetwork(3, 16, [2, 4, 1]);
    const input = Matrix.random(2, 2);
    const out0 = tchn.forward(0, input);
    const out1 = tchn.forward(1, input);
    // Check all output elements — at least one should differ
    let maxDiff = 0;
    for (let i = 0; i < out0.data.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(out0.data[i] - out1.data[i]));
    }
    assert.ok(maxDiff > 1e-6, 'Different tasks should differ');
  });

  it('interpolation produces valid output', () => {
    const tchn = new TaskConditionedHyperNetwork(2, 8, [2, 3, 1]);
    const network = tchn.interpolate(0, 1, 0.5);
    const input = Matrix.random(2, 2);
    const output = network.forward(input);
    assert.equal(output.rows, 2);
    assert.ok(Number.isFinite(output.get(0, 0)));
  });

  it('throws on invalid task', () => {
    const tchn = new TaskConditionedHyperNetwork(3, 4, [2, 1]);
    assert.throws(() => tchn.forward(5, Matrix.random(1, 2)));
  });

  it('getTaskNetwork returns functional network', () => {
    const tchn = new TaskConditionedHyperNetwork(3, 4, [2, 3, 1]);
    const net = tchn.getTaskNetwork(1);
    assert.ok(net.layers.length === 2);
    const output = net.forward(Matrix.random(1, 2));
    assert.ok(Number.isFinite(output.get(0, 0)));
  });
});

describe('FiLM', () => {
  it('modulates features', () => {
    const film = new FiLM(4, 3);
    const features = [1, 2, 3];
    const condition = [1, 0, 0, 0];
    const result = film.modulate(features, condition);
    assert.equal(result.length, 3);
    assert.ok(result.every(Number.isFinite));
  });

  it('identity conditioning preserves features', () => {
    const film = new FiLM(2, 3);
    // Set gamma bias to 1, beta bias to 0, weights to 0
    film.gammaWeights = [[0, 0], [0, 0], [0, 0]];
    film.gammaBias = [1, 1, 1];
    film.betaWeights = [[0, 0], [0, 0], [0, 0]];
    film.betaBias = [0, 0, 0];

    const features = [5, 10, 15];
    const result = film.modulate(features, [1, 1]);
    for (let i = 0; i < 3; i++) {
      assert.ok(approx(result[i], features[i], 0.001));
    }
  });

  it('different conditions produce different modulations', () => {
    const film = new FiLM(4, 3);
    const features = [1, 2, 3];
    const r1 = film.modulate(features, [1, 0, 0, 0]);
    const r2 = film.modulate(features, [0, 0, 0, 1]);
    let different = false;
    for (let i = 0; i < 3; i++) {
      if (Math.abs(r1[i] - r2[i]) > 0.001) different = true;
    }
    assert.ok(different, 'Different conditions should give different modulations');
  });
});

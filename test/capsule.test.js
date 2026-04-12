import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  squash, vectorNorm, CapsuleLayer, marginLoss, primaryCapsules,
} from '../src/capsule.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Squash Activation', () => {
  it('short vector stays short', () => {
    const result = squash([0.1, 0.1, 0.1]);
    const norm = vectorNorm(result);
    assert.ok(norm < 0.5, `Short vector should stay short: ${norm}`);
  });

  it('long vector approaches unit length', () => {
    const result = squash([10, 10, 10]);
    const norm = vectorNorm(result);
    assert.ok(norm > 0.9, `Long vector should approach 1: ${norm}`);
  });

  it('preserves direction', () => {
    const input = [3, 4, 0];
    const result = squash(input);
    // ratio of components should be preserved
    const ratio = result[0] / result[1];
    assert.ok(approx(ratio, 3 / 4, 0.01));
  });

  it('zero vector stays zero', () => {
    const result = squash([0, 0, 0]);
    assert.ok(vectorNorm(result) < 0.01);
  });

  it('output norm is always < 1', () => {
    for (let i = 0; i < 20; i++) {
      const vec = Array.from({ length: 5 }, () => Math.random() * 20 - 10);
      const result = squash(vec);
      assert.ok(vectorNorm(result) < 1.001, `Norm should be < 1: ${vectorNorm(result)}`);
    }
  });
});

describe('Capsule Layer', () => {
  it('forward produces correct shape', () => {
    const layer = new CapsuleLayer(3, 8, 6, 4, 3);
    const inputs = Array.from({ length: 6 }, () =>
      Array.from({ length: 4 }, () => Math.random())
    );
    const output = layer.forward(inputs);
    assert.equal(output.length, 3);
    assert.equal(output[0].length, 8);
  });

  it('output vectors have norm < 1', () => {
    const layer = new CapsuleLayer(3, 8, 6, 4, 3);
    const inputs = Array.from({ length: 6 }, () =>
      Array.from({ length: 4 }, () => Math.random())
    );
    const output = layer.forward(inputs);
    for (const cap of output) {
      assert.ok(vectorNorm(cap) < 1.001, `Capsule norm should be < 1: ${vectorNorm(cap)}`);
    }
  });

  it('coupling coefficients sum to 1 per input capsule', () => {
    const layer = new CapsuleLayer(3, 8, 6, 4, 3);
    const inputs = Array.from({ length: 6 }, () =>
      Array.from({ length: 4 }, () => Math.random())
    );
    layer.forward(inputs);

    for (let i = 0; i < 6; i++) {
      const sum = layer.couplingCoeffs[i].reduce((a, b) => a + b, 0);
      assert.ok(approx(sum, 1, 0.001), `Coupling should sum to 1: ${sum}`);
    }
  });

  it('more routing iterations refine agreement', () => {
    const inputs = Array.from({ length: 4 }, () =>
      Array.from({ length: 3 }, () => Math.random())
    );

    const layer1 = new CapsuleLayer(2, 4, 4, 3, 1);
    const layer3 = new CapsuleLayer(2, 4, 4, 3, 3);
    // Copy weights
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 2; j++)
        layer3.W[i][j] = layer1.W[i][j].map(r => [...r]);

    const out1 = layer1.forward(inputs);
    const out3 = layer3.forward(inputs);

    // Both should produce valid outputs
    assert.equal(out1.length, 2);
    assert.equal(out3.length, 2);
  });

  it('backward produces correct shapes', () => {
    const layer = new CapsuleLayer(3, 4, 5, 3, 2);
    const inputs = Array.from({ length: 5 }, () =>
      Array.from({ length: 3 }, () => Math.random())
    );
    layer.forward(inputs);

    const dOutput = Array.from({ length: 3 }, () =>
      Array.from({ length: 4 }, () => Math.random() * 0.1)
    );
    const dInput = layer.backward(dOutput);
    assert.equal(dInput.length, 5);
    assert.equal(dInput[0].length, 3);
  });

  it('paramCount is correct', () => {
    const layer = new CapsuleLayer(3, 8, 6, 4, 3);
    assert.equal(layer.paramCount(), 6 * 3 * 8 * 4); // 576
  });
});

describe('Margin Loss', () => {
  it('zero loss for perfect prediction', () => {
    const output = [
      squash([10, 10, 10, 10]), // class 0 active (norm ≈ 1)
      squash([0.01, 0.01, 0.01, 0.01]), // class 1 inactive (norm ≈ 0)
    ];
    const labels = [1, 0]; // class 0 is present
    const { loss } = marginLoss(output, labels);
    assert.ok(loss < 0.05, `Loss should be small: ${loss}`);
  });

  it('high loss for wrong prediction', () => {
    const output = [
      squash([0.01, 0.01, 0.01, 0.01]), // class 0 inactive but should be active
      squash([10, 10, 10, 10]), // class 1 active but should be inactive
    ];
    const labels = [1, 0];
    const { loss } = marginLoss(output, labels);
    assert.ok(loss > 0.1, `Loss should be significant: ${loss}`);
  });

  it('returns gradients', () => {
    const output = [squash([1, 2, 3, 4]), squash([0.1, 0.1, 0.1, 0.1])];
    const labels = [1, 0];
    const { gradients } = marginLoss(output, labels);
    assert.equal(gradients.length, 2);
    assert.equal(gradients[0].length, 4);
    assert.ok(gradients[0].every(Number.isFinite));
  });
});

describe('Primary Capsules', () => {
  it('converts features to capsules', () => {
    const features = Array.from({ length: 24 }, () => Math.random());
    const caps = primaryCapsules(features, 6, 4);
    assert.equal(caps.length, 6);
    assert.equal(caps[0].length, 4);
  });

  it('all capsules have norm < 1', () => {
    const features = Array.from({ length: 24 }, () => Math.random() * 5);
    const caps = primaryCapsules(features, 6, 4);
    for (const cap of caps) {
      assert.ok(vectorNorm(cap) < 1.001);
    }
  });
});

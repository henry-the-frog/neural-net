import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  PlanarFlow, AffineCouplingLayer, ActNorm, NormalizingFlow,
} from '../src/normalizing-flows.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Planar Flow', () => {
  it('forward produces output and logDetJ', () => {
    const flow = new PlanarFlow(3);
    const result = flow.forward([1, 2, 3]);
    assert.equal(result.z.length, 3);
    assert.ok(Number.isFinite(result.logDetJ));
  });

  it('approximate inverse roundtrips', () => {
    const flow = new PlanarFlow(3);
    const input = [0.5, -0.3, 0.8];
    const { z } = flow.forward(input);
    const recovered = flow.inverse(z, 50);
    for (let i = 0; i < 3; i++) {
      assert.ok(approx(recovered[i], input[i], 0.1),
        `Planar inverse mismatch at ${i}: ${recovered[i].toFixed(4)} vs ${input[i]}`);
    }
  });

  it('logDetJ is finite for many inputs', () => {
    const flow = new PlanarFlow(4);
    for (let trial = 0; trial < 20; trial++) {
      const input = Array.from({ length: 4 }, () => Math.random() * 4 - 2);
      const { logDetJ } = flow.forward(input);
      assert.ok(Number.isFinite(logDetJ), `logDetJ should be finite: ${logDetJ}`);
    }
  });
});

describe('Affine Coupling Layer', () => {
  it('forward produces correct shape', () => {
    const layer = new AffineCouplingLayer(4);
    const result = layer.forward([1, 2, 3, 4]);
    assert.equal(result.z.length, 4);
    assert.ok(Number.isFinite(result.logDetJ));
  });

  it('exact inverse roundtrips', () => {
    const layer = new AffineCouplingLayer(4);
    const input = [0.5, -0.3, 0.8, -0.1];
    const { z } = layer.forward(input);
    const recovered = layer.inverse(z);
    for (let i = 0; i < 4; i++) {
      assert.ok(approx(recovered[i], input[i], 0.001),
        `Affine inverse mismatch at ${i}: ${recovered[i].toFixed(4)} vs ${input[i]}`);
    }
  });

  it('first half is unchanged', () => {
    const layer = new AffineCouplingLayer(4, 2);
    const input = [1, 2, 3, 4];
    const { z } = layer.forward(input);
    assert.equal(z[0], 1, 'First element should be unchanged');
    assert.equal(z[1], 2, 'Second element should be unchanged');
  });

  it('inverse roundtrips 20 random inputs', () => {
    const layer = new AffineCouplingLayer(6);
    for (let trial = 0; trial < 20; trial++) {
      const input = Array.from({ length: 6 }, () => Math.random() * 4 - 2);
      const { z } = layer.forward(input);
      const recovered = layer.inverse(z);
      for (let i = 0; i < 6; i++) {
        assert.ok(approx(recovered[i], input[i], 0.001),
          `Trial ${trial}, dim ${i}: ${recovered[i].toFixed(4)} vs ${input[i].toFixed(4)}`);
      }
    }
  });
});

describe('ActNorm', () => {
  it('initializes from data', () => {
    const an = new ActNorm(3);
    const batch = Array.from({ length: 100 }, () =>
      Array.from({ length: 3 }, () => Math.random() * 10 + 5)
    );
    an.initialize(batch);
    assert.ok(an.initialized);
  });

  it('normalizes data to ~zero mean, ~unit variance', () => {
    const an = new ActNorm(2);
    const batch = Array.from({ length: 100 }, () => [10 + Math.random(), 20 + Math.random()]);
    an.initialize(batch);

    // Transform a sample
    const { z } = an.forward([10.5, 20.5]);
    // Should be close to 0
    assert.ok(Math.abs(z[0]) < 5, `Should be near zero: ${z[0]}`);
    assert.ok(Math.abs(z[1]) < 5, `Should be near zero: ${z[1]}`);
  });

  it('inverse roundtrips', () => {
    const an = new ActNorm(3);
    an.scale = [2, 0.5, 1.5];
    an.bias = [-1, 2, 0];
    const input = [5, -3, 7];
    const { z } = an.forward(input);
    const recovered = an.inverse(z);
    for (let i = 0; i < 3; i++) {
      assert.ok(approx(recovered[i], input[i], 0.001));
    }
  });
});

describe('Normalizing Flow', () => {
  it('forward and inverse roundtrip', () => {
    const flow = new NormalizingFlow([
      new AffineCouplingLayer(4),
      new AffineCouplingLayer(4, 3), // Different split
      new AffineCouplingLayer(4),
    ]);

    const input = [1, -0.5, 0.3, 2];
    const { z } = flow.forward(input);
    const recovered = flow.inverse(z);

    for (let i = 0; i < 4; i++) {
      assert.ok(approx(recovered[i], input[i], 0.001),
        `Flow roundtrip at ${i}: ${recovered[i].toFixed(4)} vs ${input[i]}`);
    }
  });

  it('logLikelihood is finite', () => {
    const flow = new NormalizingFlow([
      new AffineCouplingLayer(4),
      new AffineCouplingLayer(4, 3),
    ]);
    const ll = flow.logLikelihood([1, 0, -1, 0.5]);
    assert.ok(Number.isFinite(ll), `Log-likelihood should be finite: ${ll}`);
  });

  it('sample produces valid output', () => {
    const flow = new NormalizingFlow([
      new AffineCouplingLayer(4),
    ]);
    const sample = flow.sample();
    assert.equal(sample.length, 4);
    assert.ok(sample.every(Number.isFinite));
  });

  it('roundtrips 20 random inputs through deep flow', () => {
    const flow = new NormalizingFlow([
      new AffineCouplingLayer(4),
      new ActNorm(4),
      new AffineCouplingLayer(4, 3),
      new ActNorm(4),
      new AffineCouplingLayer(4),
    ]);

    for (let trial = 0; trial < 20; trial++) {
      const input = Array.from({ length: 4 }, () => Math.random() * 2 - 1);
      const { z } = flow.forward(input);
      const recovered = flow.inverse(z);
      for (let i = 0; i < 4; i++) {
        assert.ok(approx(recovered[i], input[i], 0.01),
          `Trial ${trial}, dim ${i}: ${recovered[i].toFixed(4)} vs ${input[i].toFixed(4)}`);
      }
    }
  });

  it('paramCount sums layers', () => {
    const flow = new NormalizingFlow([
      new PlanarFlow(3),
      new AffineCouplingLayer(3),
    ]);
    assert.ok(flow.paramCount() > 0);
  });
});

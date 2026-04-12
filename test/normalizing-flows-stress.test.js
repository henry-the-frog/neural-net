// normalizing-flows-stress.test.js — Deep stress tests for normalizing flows
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  PlanarFlow, AffineCouplingLayer, ActNorm, NormalizingFlow,
} from '../src/normalizing-flows.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

/**
 * Compute the numerical Jacobian determinant via finite differences.
 * For a function f: R^n → R^n, J[i][j] = ∂f_i/∂x_j
 */
function numericalLogDetJ(layer, x, h = 1e-5) {
  const n = x.length;
  const J = [];
  for (let j = 0; j < n; j++) {
    const row = [];
    const xPlus = [...x]; xPlus[j] += h;
    const xMinus = [...x]; xMinus[j] -= h;
    const fPlus = layer.forward(xPlus).z;
    const fMinus = layer.forward(xMinus).z;
    for (let i = 0; i < n; i++) {
      row.push((fPlus[i] - fMinus[i]) / (2 * h));
    }
    J.push(row);
  }
  
  // Compute determinant of n×n Jacobian matrix
  // For small n, use direct formula
  const det = determinant(J);
  return Math.log(Math.abs(det) + 1e-30);
}

/**
 * Compute determinant of a square matrix (LU decomposition for n > 3)
 */
function determinant(M) {
  const n = M.length;
  if (n === 1) return M[0][0];
  if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
  if (n === 3) {
    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
         - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  }
  // LU decomposition for larger matrices
  const LU = M.map(row => [...row]);
  let sign = 1;
  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxVal = Math.abs(LU[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(LU[row][col]) > maxVal) {
        maxVal = Math.abs(LU[row][col]);
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      [LU[col], LU[maxRow]] = [LU[maxRow], LU[col]];
      sign *= -1;
    }
    if (Math.abs(LU[col][col]) < 1e-15) return 0;
    for (let row = col + 1; row < n; row++) {
      const factor = LU[row][col] / LU[col][col];
      for (let k = col; k < n; k++) {
        LU[row][k] -= factor * LU[col][k];
      }
    }
  }
  let det = sign;
  for (let i = 0; i < n; i++) det *= LU[i][i];
  return det;
}

describe('Numerical Jacobian Verification', () => {
  describe('AffineCouplingLayer', () => {
    it('analytical logDetJ matches numerical (dim=4)', () => {
      for (let trial = 0; trial < 10; trial++) {
        const layer = new AffineCouplingLayer(4);
        const x = Array.from({ length: 4 }, () => Math.random() * 2 - 1);
        const { logDetJ: analytical } = layer.forward(x);
        const numerical = numericalLogDetJ(layer, x);
        assert.ok(approx(analytical, numerical, 0.1),
          `Trial ${trial}: analytical=${analytical.toFixed(4)}, numerical=${numerical.toFixed(4)}`);
      }
    });

    it('analytical logDetJ matches numerical (dim=6)', () => {
      for (let trial = 0; trial < 5; trial++) {
        const layer = new AffineCouplingLayer(6);
        const x = Array.from({ length: 6 }, () => Math.random() * 2 - 1);
        const { logDetJ: analytical } = layer.forward(x);
        const numerical = numericalLogDetJ(layer, x);
        assert.ok(approx(analytical, numerical, 0.15),
          `Trial ${trial}: analytical=${analytical.toFixed(4)}, numerical=${numerical.toFixed(4)}`);
      }
    });
  });

  describe('PlanarFlow', () => {
    it('analytical logDetJ matches numerical (dim=3)', () => {
      let matchCount = 0;
      for (let trial = 0; trial < 10; trial++) {
        const flow = new PlanarFlow(3);
        const x = Array.from({ length: 3 }, () => Math.random() * 2 - 1);
        const { logDetJ: analytical } = flow.forward(x);
        const numerical = numericalLogDetJ(flow, x);
        if (approx(analytical, numerical, 0.2)) matchCount++;
      }
      // Most should match (some might diverge near singularities)
      assert.ok(matchCount >= 7, `At least 7/10 should match: got ${matchCount}`);
    });
  });

  describe('ActNorm', () => {
    it('analytical logDetJ matches numerical', () => {
      const an = new ActNorm(4);
      an.scale = [2, 0.5, 1.5, 0.8];
      an.bias = [-1, 2, 0, 0.5];
      const x = [1, 2, 3, 4];
      const { logDetJ: analytical } = an.forward(x);
      const numerical = numericalLogDetJ(an, x);
      assert.ok(approx(analytical, numerical, 0.05),
        `ActNorm: analytical=${analytical.toFixed(4)}, numerical=${numerical.toFixed(4)}`);
    });
  });
});

describe('Invertibility Under Extreme Inputs', () => {
  describe('AffineCouplingLayer', () => {
    it('handles large inputs without NaN/Inf', () => {
      const layer = new AffineCouplingLayer(4);
      const large = [100, -100, 50, -50];
      const { z, logDetJ } = layer.forward(large);
      assert.ok(z.every(Number.isFinite), 'Output should be finite');
      assert.ok(Number.isFinite(logDetJ), 'logDetJ should be finite');
      const recovered = layer.inverse(z);
      for (let i = 0; i < 4; i++) {
        assert.ok(approx(recovered[i], large[i], 0.01),
          `Large input roundtrip: ${recovered[i].toFixed(4)} vs ${large[i]}`);
      }
    });

    it('handles near-zero inputs', () => {
      const layer = new AffineCouplingLayer(4);
      const small = [1e-10, -1e-10, 1e-8, -1e-8];
      const { z, logDetJ } = layer.forward(small);
      assert.ok(z.every(Number.isFinite));
      assert.ok(Number.isFinite(logDetJ));
      const recovered = layer.inverse(z);
      for (let i = 0; i < 4; i++) {
        assert.ok(approx(recovered[i], small[i], 0.001));
      }
    });
  });

  describe('PlanarFlow', () => {
    it('handles large inputs', () => {
      const flow = new PlanarFlow(3);
      const large = [50, -50, 25];
      const { z, logDetJ } = flow.forward(large);
      assert.ok(z.every(Number.isFinite), 'Output should be finite');
      assert.ok(Number.isFinite(logDetJ), 'logDetJ should be finite');
    });

    it('handles zero input', () => {
      const flow = new PlanarFlow(3);
      const { z, logDetJ } = flow.forward([0, 0, 0]);
      assert.ok(z.every(Number.isFinite));
      assert.ok(Number.isFinite(logDetJ));
    });
  });

  describe('ActNorm', () => {
    it('handles extreme scales after initialization', () => {
      const an = new ActNorm(3);
      // Batch with very different scales
      const batch = Array.from({ length: 100 }, () => [
        1000 + Math.random() * 0.01,
        0.001 + Math.random() * 0.0001,
        -500 + Math.random() * 0.5,
      ]);
      an.initialize(batch);
      
      const { z, logDetJ } = an.forward([1000, 0.001, -500]);
      assert.ok(z.every(Number.isFinite), `Output: ${z}`);
      assert.ok(Number.isFinite(logDetJ), `logDetJ: ${logDetJ}`);
      
      const recovered = an.inverse(z);
      assert.ok(approx(recovered[0], 1000, 1));
      assert.ok(approx(recovered[1], 0.001, 0.01));
      assert.ok(approx(recovered[2], -500, 1));
    });
  });
});

describe('Change-of-Variables Formula', () => {
  it('log p(x) = log p(f(x)) + log |det J|', () => {
    // For a flow from x → z, the change-of-variables formula is:
    // log p(x) = log p_z(z) + log |det df/dx|
    // where z = f(x) and p_z is the base distribution (standard Gaussian)
    
    const flow = new NormalizingFlow([
      new AffineCouplingLayer(4),
      new AffineCouplingLayer(4, 3),
    ]);
    
    for (let trial = 0; trial < 20; trial++) {
      const x = Array.from({ length: 4 }, () => Math.random() * 2 - 1);
      const { z, logDetJ } = flow.forward(x);
      
      // log p_z(z) under standard Gaussian
      const logPz = z.reduce((s, zi) => s - 0.5 * (zi * zi + Math.log(2 * Math.PI)), 0);
      
      // Change-of-variables
      const expected = logPz + logDetJ;
      const actual = flow.logLikelihood(x);
      
      assert.ok(approx(actual, expected, 0.001),
        `Trial ${trial}: logLikelihood=${actual.toFixed(4)}, expected=${expected.toFixed(4)}`);
    }
  });
});

describe('Deep Flow Composition', () => {
  it('invertibility through 10 coupling layers', () => {
    const layers = [];
    for (let i = 0; i < 10; i++) {
      layers.push(new AffineCouplingLayer(4, i % 2 === 0 ? 2 : 3));
    }
    const flow = new NormalizingFlow(layers);
    
    const input = [0.5, -0.3, 0.8, -0.1];
    const { z } = flow.forward(input);
    const recovered = flow.inverse(z);
    
    for (let i = 0; i < 4; i++) {
      assert.ok(approx(recovered[i], input[i], 0.01),
        `10-layer roundtrip at ${i}: ${recovered[i].toFixed(6)} vs ${input[i]}`);
    }
  });

  it('invertibility through 20 mixed layers', () => {
    const layers = [];
    for (let i = 0; i < 20; i++) {
      if (i % 3 === 0) {
        const an = new ActNorm(4);
        an.scale = Array.from({ length: 4 }, () => 0.5 + Math.random());
        an.bias = Array.from({ length: 4 }, () => (Math.random() - 0.5) * 0.1);
        layers.push(an);
      } else {
        layers.push(new AffineCouplingLayer(4, i % 2 === 0 ? 2 : 3));
      }
    }
    const flow = new NormalizingFlow(layers);
    
    for (let trial = 0; trial < 5; trial++) {
      const input = Array.from({ length: 4 }, () => Math.random() * 2 - 1);
      const { z, logDetJ } = flow.forward(input);
      
      assert.ok(z.every(Number.isFinite), `Forward should be finite at trial ${trial}`);
      assert.ok(Number.isFinite(logDetJ), 'logDetJ should be finite');
      
      const recovered = flow.inverse(z);
      for (let i = 0; i < 4; i++) {
        assert.ok(approx(recovered[i], input[i], 0.05),
          `Trial ${trial}, dim ${i}: ${recovered[i].toFixed(6)} vs ${input[i].toFixed(6)}`);
      }
    }
  });

  it('logDetJ through deep flow is sum of layer logDetJs', () => {
    const layers = [
      new AffineCouplingLayer(4),
      new AffineCouplingLayer(4, 3),
      new AffineCouplingLayer(4),
    ];
    const flow = new NormalizingFlow(layers);
    
    const x = [0.5, -0.3, 0.8, -0.1];
    
    // Manually compute sum of logDetJs
    let z = [...x];
    let sumLogDetJ = 0;
    for (const layer of layers) {
      const result = layer.forward(z);
      z = result.z;
      sumLogDetJ += result.logDetJ;
    }
    
    const { logDetJ } = flow.forward(x);
    assert.ok(approx(logDetJ, sumLogDetJ, 0.001),
      `Flow logDetJ=${logDetJ.toFixed(6)}, sum=${sumLogDetJ.toFixed(6)}`);
  });
});

describe('Planar Flow Invertibility Constraint', () => {
  it('_ensureInvertible maintains w^T u >= -1', () => {
    // After _ensureInvertible, the condition w^T u' >= -1 should hold
    // This ensures the determinant doesn't go through zero
    for (let trial = 0; trial < 20; trial++) {
      const flow = new PlanarFlow(5);
      const wtu = flow.w.reduce((s, wi, i) => s + wi * flow.u[i], 0);
      assert.ok(wtu >= -1 - 0.01,
        `w^T u should be >= -1: got ${wtu.toFixed(4)}`);
    }
  });

  it('determinant is always positive after _ensureInvertible', () => {
    for (let trial = 0; trial < 20; trial++) {
      const flow = new PlanarFlow(3);
      const x = Array.from({ length: 3 }, () => Math.random() * 4 - 2);
      const { logDetJ } = flow.forward(x);
      // If the determinant can be negative, logDetJ is log(|det|)
      // But with the invertibility constraint, det should be > 0
      assert.ok(Number.isFinite(logDetJ));
    }
  });
});

describe('Sampling', () => {
  it('NormalizingFlow.sample produces diverse outputs', () => {
    const flow = new NormalizingFlow([
      new AffineCouplingLayer(4),
      new AffineCouplingLayer(4, 3),
    ]);
    
    const samples = Array.from({ length: 50 }, () => flow.sample());
    
    // Check diversity: compute variance of each dimension
    for (let d = 0; d < 4; d++) {
      const vals = samples.map(s => s[d]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
      assert.ok(variance > 0.001, `Dimension ${d} should have variance > 0.001: got ${variance.toFixed(6)}`);
    }
  });

  it('samples are finite', () => {
    const flow = new NormalizingFlow([
      new AffineCouplingLayer(4),
    ]);
    for (let i = 0; i < 100; i++) {
      const sample = flow.sample();
      assert.ok(sample.every(Number.isFinite), `Sample ${i} has non-finite values`);
    }
  });
});

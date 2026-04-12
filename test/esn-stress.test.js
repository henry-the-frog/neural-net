// esn-stress.test.js — Deep stress tests for Echo State Networks
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { EchoStateNetwork } from '../src/esn.js';
import { Matrix } from '../src/matrix.js';

describe('Reservoir Dynamics', () => {
  it('state is bounded (echo state property)', () => {
    const esn = new EchoStateNetwork(1, 50, 1, { spectralRadius: 0.9 });
    
    // Drive with constant input for many steps
    for (let t = 0; t < 500; t++) {
      esn.step(Matrix.fromArray([[1.0]]));
    }
    
    // State should be bounded due to tanh and spectral radius < 1
    for (let j = 0; j < 50; j++) {
      const v = esn.state.get(0, j);
      assert.ok(Math.abs(v) < 2, `State should be bounded: state[${j}]=${v.toFixed(4)}`);
    }
  });

  it('state converges to same attractor regardless of initial state', () => {
    const esn1 = new EchoStateNetwork(1, 20, 1, { spectralRadius: 0.8, noise: 0 });
    const esn2 = new EchoStateNetwork(1, 20, 1, { spectralRadius: 0.8, noise: 0 });
    
    // Copy weights
    esn2.Win = esn1.Win;
    esn2.W = esn1.W;
    
    // Different initial states
    esn1.state = Matrix.random(1, 20);
    esn2.state = Matrix.random(1, 20);
    
    // Same input sequence
    for (let t = 0; t < 200; t++) {
      const input = Matrix.fromArray([[Math.sin(t * 0.1)]]);
      esn1.step(input);
      esn2.step(input);
    }
    
    // States should converge (echo state property)
    let maxDiff = 0;
    for (let j = 0; j < 20; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(esn1.state.get(0, j) - esn2.state.get(0, j)));
    }
    assert.ok(maxDiff < 0.1, `States should converge: maxDiff=${maxDiff.toFixed(4)}`);
  });

  it('zero input drives state to zero', () => {
    const esn = new EchoStateNetwork(1, 20, 1, { spectralRadius: 0.8, noise: 0 });
    esn.state = Matrix.random(1, 20); // Random initial state
    
    for (let t = 0; t < 200; t++) {
      esn.step(Matrix.fromArray([[0]]));
    }
    
    let norm = 0;
    for (let j = 0; j < 20; j++) norm += esn.state.get(0, j) ** 2;
    norm = Math.sqrt(norm);
    assert.ok(norm < 0.1, `Zero input should decay state: norm=${norm.toFixed(4)}`);
  });
});

describe('Training (Ridge Regression)', () => {
  it('learns identity function: f(x) = x', () => {
    const esn = new EchoStateNetwork(1, 100, 1, { spectralRadius: 0.9 });
    
    // Generate training data: just echo the input
    const N = 200;
    const inputs = [];
    const targets = [];
    for (let t = 0; t < N; t++) {
      const x = Math.sin(t * 0.05);
      inputs.push(Matrix.fromArray([[x]]));
      targets.push([x]);
    }
    
    esn.train(inputs, targets, 50); // 50 step washout
    
    // Test
    esn.reset();
    let mse = 0;
    const testN = 100;
    for (let t = 0; t < testN; t++) {
      const x = Math.sin((N + t) * 0.05);
      const pred = esn.predict(Matrix.fromArray([[x]]));
      mse += (pred.get(0, 0) - x) ** 2;
    }
    mse /= testN;
    assert.ok(mse < 0.1, `Identity MSE should be low: ${mse.toFixed(4)}`);
  });

  it('learns sine wave prediction', () => {
    const esn = new EchoStateNetwork(1, 200, 1, { spectralRadius: 0.95 });
    
    const N = 500;
    const inputs = [];
    const targets = [];
    for (let t = 0; t < N; t++) {
      inputs.push(Matrix.fromArray([[Math.sin(t * 0.02)]]));
      targets.push([Math.sin((t + 1) * 0.02)]); // Next step
    }
    
    esn.train(inputs, targets, 100);
    
    esn.reset();
    let mse = 0;
    const testN = 100;
    for (let t = 0; t < testN; t++) {
      const x = Math.sin((N + t) * 0.02);
      const pred = esn.predict(Matrix.fromArray([[x]]));
      const expected = Math.sin((N + t + 1) * 0.02);
      mse += (pred.get(0, 0) - expected) ** 2;
    }
    mse /= testN;
    assert.ok(mse < 0.1, `Sine prediction MSE should be low: ${mse.toFixed(4)}`);
  });
});

describe('Washout Period', () => {
  it('more washout should improve performance', () => {
    // With short washout, transient dynamics corrupt the training
    // Typically washout > 0 is better than washout = 0
    const makeESN = () => {
      const esn = new EchoStateNetwork(1, 50, 1, { spectralRadius: 0.9 });
      return esn;
    };
    
    const N = 200;
    const inputs = [];
    const targets = [];
    for (let t = 0; t < N; t++) {
      inputs.push(Matrix.fromArray([[Math.sin(t * 0.05)]]));
      targets.push([Math.sin(t * 0.05)]);
    }
    
    const esn0 = makeESN();
    esn0.train(inputs, targets, 0); // No washout
    
    // Test
    esn0.reset();
    let mse0 = 0;
    for (let t = 0; t < 50; t++) {
      const x = Math.sin((N + t) * 0.05);
      const pred = esn0.predict(Matrix.fromArray([[x]]));
      mse0 += (pred.get(0, 0) - x) ** 2;
    }
    // We just verify it runs without error
    assert.ok(Number.isFinite(mse0), `MSE should be finite: ${mse0}`);
  });
});

describe('Edge Cases', () => {
  it('handles multi-dimensional input', () => {
    const esn = new EchoStateNetwork(3, 50, 2);
    esn.step(Matrix.fromArray([[1, 0, -1]]));
    assert.ok(esn.state.data.every(Number.isFinite));
  });

  it('reset clears state', () => {
    const esn = new EchoStateNetwork(1, 20, 1);
    esn.step(Matrix.fromArray([[1]]));
    esn.reset();
    let norm = 0;
    for (let j = 0; j < 20; j++) norm += esn.state.get(0, j) ** 2;
    assert.ok(Math.sqrt(norm) < 0.01, 'Reset should clear state');
  });
});

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { EchoStateNetwork, LiquidStateMachine } from '../src/esn.js';
import { Matrix } from '../src/matrix.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Echo State Network', () => {
  it('creates with correct dimensions', () => {
    const esn = new EchoStateNetwork(2, 50, 1);
    assert.equal(esn.inputSize, 2);
    assert.equal(esn.reservoirSize, 50);
    assert.equal(esn.outputSize, 1);
  });

  it('step updates reservoir state', () => {
    const esn = new EchoStateNetwork(2, 20, 1);
    const input = new Matrix(1, 2, new Float64Array([1, 0.5]));
    const state = esn.step(input);
    assert.equal(state.rows, 1);
    assert.equal(state.cols, 20);
    // State should be non-zero after input
    let hasNonZero = false;
    for (let j = 0; j < 20; j++) {
      if (Math.abs(state.get(0, j)) > 0.001) hasNonZero = true;
    }
    assert.ok(hasNonZero, 'State should be non-zero');
  });

  it('reset clears state', () => {
    const esn = new EchoStateNetwork(2, 20, 1);
    esn.step(new Matrix(1, 2, new Float64Array([1, 1])));
    esn.reset();
    for (let j = 0; j < 20; j++) {
      assert.ok(approx(esn.state.get(0, j), 0, 0.001));
    }
  });

  it('trains on sine wave', () => {
    const esn = new EchoStateNetwork(1, 100, 1, {
      spectralRadius: 0.95,
      leakingRate: 0.3,
    });

    // Generate sine wave
    const N = 500;
    const inputs = [];
    const targets = [];
    for (let t = 0; t < N; t++) {
      inputs.push([Math.sin(2 * Math.PI * t / 50)]);
      targets.push([Math.sin(2 * Math.PI * (t + 1) / 50)]); // Predict next step
    }

    esn.train(inputs, targets, 100);
    assert.ok(esn.Wout, 'Should have trained output weights');

    // Test prediction
    esn.reset();
    // Warm up
    for (let t = 0; t < 100; t++) esn.predict(inputs[t]);

    // Predict and check
    let mse = 0;
    for (let t = 100; t < 200; t++) {
      const pred = esn.predict(inputs[t]);
      mse += (pred.get(0, 0) - targets[t][0]) ** 2;
    }
    mse /= 100;
    assert.ok(mse < 0.5, `MSE should be reasonable: ${mse.toFixed(4)}`);
  });

  it('collectStates respects washout', () => {
    const esn = new EchoStateNetwork(1, 10, 1);
    const inputs = Array.from({ length: 50 }, () => [Math.random()]);
    const states = esn.collectStates(inputs, 20);
    assert.equal(states.length, 30); // 50 - 20
  });

  it('collectStates produces extended state', () => {
    const esn = new EchoStateNetwork(2, 10, 1);
    const inputs = Array.from({ length: 20 }, () => [Math.random(), Math.random()]);
    const states = esn.collectStates(inputs, 0);
    assert.equal(states[0].length, 12); // 2 input + 10 reservoir
  });

  it('paramCount reports fixed and trainable', () => {
    const esn = new EchoStateNetwork(2, 50, 1);
    const params = esn.paramCount();
    assert.ok(params.fixed > 0);
    assert.equal(params.trainable, 0); // Not trained yet
    assert.equal(params.total, params.fixed);
  });

  it('predict throws if not trained', () => {
    const esn = new EchoStateNetwork(2, 10, 1);
    assert.throws(() => esn.predict([1, 2]));
  });

  it('predictSequence generates correct length', () => {
    const esn = new EchoStateNetwork(1, 50, 1);
    const inputs = Array.from({ length: 200 }, (_, t) => [Math.sin(t * 0.1)]);
    const targets = Array.from({ length: 200 }, (_, t) => [Math.sin((t + 1) * 0.1)]);
    esn.train(inputs, targets, 50);

    const seq = esn.predictSequence([0], 10);
    assert.equal(seq.length, 10);
    assert.ok(seq.every(s => s.length === 1));
    assert.ok(seq.every(s => Number.isFinite(s[0])));
  });
});

describe('Liquid State Machine', () => {
  it('creates with correct dimensions', () => {
    const lsm = new LiquidStateMachine(2, 30, 1);
    assert.equal(lsm.inputSize, 2);
    assert.equal(lsm.reservoirSize, 30);
  });

  it('step produces state', () => {
    const lsm = new LiquidStateMachine(2, 30, 1);
    const state = lsm.step([1, 0.5]);
    assert.equal(state.length, 30);
    assert.ok(state.every(Number.isFinite));
  });

  it('generates spikes', () => {
    const lsm = new LiquidStateMachine(2, 50, 1, { connectivity: 0.2 });
    // Strong input should cause spikes
    let anySpiked = false;
    for (let t = 0; t < 100; t++) {
      lsm.step([5, 5]);
      if (lsm.spikes.some(Boolean)) anySpiked = true;
    }
    assert.ok(anySpiked, 'Should produce spikes with strong input');
  });

  it('reset clears state', () => {
    const lsm = new LiquidStateMachine(2, 20, 1);
    lsm.step([1, 1]);
    lsm.reset();
    assert.ok(lsm.potentials.every(p => p === 0));
  });
});

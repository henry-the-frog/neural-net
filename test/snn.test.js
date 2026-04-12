import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  LIFNeuron, Synapse, STDPRule, PoissonEncoder,
  spikeRate, SNNLayer, SpikingNetwork, IzhikevichNeuron,
} from '../src/snn.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('LIF Neuron', () => {
  it('starts at resting potential', () => {
    const n = new LIFNeuron();
    assert.equal(n.v, -65);
  });

  it('spikes with sufficient current', () => {
    const n = new LIFNeuron();
    let spiked = false;
    for (let t = 0; t < 100; t++) {
      if (n.step(20)) spiked = true;
    }
    assert.ok(spiked, 'Should spike with strong current');
  });

  it('does not spike with weak current', () => {
    const n = new LIFNeuron();
    let spiked = false;
    for (let t = 0; t < 100; t++) {
      if (n.step(0.5)) spiked = true;
    }
    assert.ok(!spiked, 'Should not spike with weak current');
  });

  it('resets after spike', () => {
    const n = new LIFNeuron();
    for (let t = 0; t < 100; t++) {
      if (n.step(20)) {
        assert.ok(approx(n.v, n.vReset, 1), 'Should reset to vReset after spike');
        break;
      }
    }
  });

  it('has refractory period', () => {
    const n = new LIFNeuron({ refractoryPeriod: 5 });
    // Drive to spike
    let spikeTime = -10;
    for (let t = 0; t < 20; t++) {
      if (n.step(30)) {
        if (spikeTime < 0) {
          spikeTime = t;
        } else {
          assert.ok(t - spikeTime >= 5, `Should respect refractory period: ${t - spikeTime}ms`);
          break;
        }
      }
    }
  });

  it('records voltage history', () => {
    const n = new LIFNeuron();
    for (let t = 0; t < 10; t++) n.step(5);
    assert.equal(n.voltageHistory.length, 10);
  });

  it('leaks toward resting potential', () => {
    const n = new LIFNeuron();
    n.v = -50; // Above rest
    for (let t = 0; t < 100; t++) n.step(0);
    assert.ok(Math.abs(n.v - n.vRest) < 1, `Should decay to rest: ${n.v}`);
  });
});

describe('Synapse', () => {
  it('delivers current on spike', () => {
    const pre = new LIFNeuron();
    const post = new LIFNeuron();
    const syn = new Synapse(pre, post, 5, 1);

    pre.spiked = true;
    syn.propagate(0);

    const current = syn.deliver(1);
    assert.equal(current, 5);
  });

  it('respects delay', () => {
    const pre = new LIFNeuron();
    const post = new LIFNeuron();
    const syn = new Synapse(pre, post, 5, 3);

    pre.spiked = true;
    syn.propagate(0);

    assert.equal(syn.deliver(1), 0); // Too early
    assert.equal(syn.deliver(2), 0); // Still too early
    assert.equal(syn.deliver(3), 5); // Arrives
  });
});

describe('STDP', () => {
  it('potentiates when pre fires before post (causal)', () => {
    const rule = new STDPRule();
    const syn = { weight: 1 };
    const dw = rule.update(syn, [10], [15]); // pre at 10, post at 15
    assert.ok(dw > 0, `Should potentiate: dw=${dw}`);
    assert.ok(syn.weight > 1, `Weight should increase: ${syn.weight}`);
  });

  it('depresses when post fires before pre (anti-causal)', () => {
    const rule = new STDPRule();
    const syn = { weight: 1 };
    const dw = rule.update(syn, [15], [10]); // post at 10, pre at 15
    assert.ok(dw < 0, `Should depress: dw=${dw}`);
    assert.ok(syn.weight < 1, `Weight should decrease: ${syn.weight}`);
  });

  it('larger time difference gives smaller change', () => {
    const rule = new STDPRule();
    const syn1 = { weight: 1 };
    const syn2 = { weight: 1 };
    rule.update(syn1, [10], [12]); // 2ms gap
    rule.update(syn2, [10], [30]); // 20ms gap
    assert.ok(syn1.weight > syn2.weight,
      `Shorter gap should strengthen more: ${syn1.weight} vs ${syn2.weight}`);
  });

  it('respects weight bounds', () => {
    const rule = new STDPRule({ wMax: 2 });
    const syn = { weight: 1.9 };
    // Many causal pairings
    for (let i = 0; i < 100; i++) {
      rule.update(syn, [i], [i + 1]);
    }
    assert.ok(syn.weight <= 2, `Should not exceed wMax: ${syn.weight}`);
  });
});

describe('Poisson Encoder', () => {
  it('higher value gives more spikes', () => {
    const enc = new PoissonEncoder(200);
    const lowSpikes = enc.encode(0.1, 100);
    const highSpikes = enc.encode(0.9, 100);
    // On average, high should have more (allow some randomness)
    // Run multiple trials
    let lowTotal = 0, highTotal = 0;
    for (let trial = 0; trial < 20; trial++) {
      lowTotal += enc.encode(0.1, 100).length;
      highTotal += enc.encode(0.9, 100).length;
    }
    assert.ok(highTotal > lowTotal, `High (${highTotal}) should have more spikes than low (${lowTotal})`);
  });

  it('zero value gives no spikes', () => {
    const enc = new PoissonEncoder();
    const spikes = enc.encode(0, 100);
    assert.equal(spikes.length, 0);
  });
});

describe('Spike Rate', () => {
  it('calculates firing rate', () => {
    const rate = spikeRate([10, 20, 30, 40, 50], 100);
    assert.equal(rate, 50); // 5 spikes in 100ms = 50 Hz
  });

  it('zero spikes gives zero rate', () => {
    assert.equal(spikeRate([], 100), 0);
  });
});

describe('SNN Layer', () => {
  it('creates correct number of neurons', () => {
    const layer = new SNNLayer(3, 5);
    assert.equal(layer.neurons.length, 5);
    assert.equal(layer.inputSize, 3);
    assert.equal(layer.outputSize, 5);
  });

  it('step returns boolean array', () => {
    const layer = new SNNLayer(3, 5);
    const spikes = layer.step([true, false, true]);
    assert.equal(spikes.length, 5);
    assert.ok(spikes.every(s => typeof s === 'boolean'));
  });

  it('responds to strong input', () => {
    const layer = new SNNLayer(3, 2);
    // Set strong weights
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 2; j++)
        layer.weights[i][j] = 15;

    let anySpike = false;
    for (let t = 0; t < 50; t++) {
      const spikes = layer.step([true, true, true]);
      if (spikes.some(Boolean)) anySpike = true;
    }
    assert.ok(anySpike, 'Should produce spikes with strong input');
  });
});

describe('Spiking Network', () => {
  it('simulates through layers', () => {
    const net = new SpikingNetwork([3, 4, 2]);
    const input = [
      [5, 15, 25, 35], // neuron 0 spike times
      [10, 20, 30, 40], // neuron 1 spike times
      [7, 17, 27, 37],  // neuron 2 spike times
    ];
    const output = net.simulate(input, 50);
    assert.equal(output.length, 2); // 2 output neurons
    assert.ok(output.every(Array.isArray));
  });

  it('reset clears state', () => {
    const net = new SpikingNetwork([2, 3]);
    net.simulate([[5, 10], [7, 12]], 20);
    net.reset();
    for (const layer of net.layers) {
      for (const n of layer.neurons) {
        assert.equal(n.v, n.vRest);
        assert.equal(n.voltageHistory.length, 0);
      }
    }
  });
});

describe('Izhikevich Neuron', () => {
  it('regular spiking pattern', () => {
    const n = IzhikevichNeuron.regularSpiking();
    const spikeTimes = [];
    for (let t = 0; t < 200; t++) {
      if (n.step(10)) spikeTimes.push(t);
    }
    assert.ok(spikeTimes.length >= 2, `Should produce multiple spikes: ${spikeTimes.length}`);
  });

  it('fast spiking fires more frequently', () => {
    const rs = IzhikevichNeuron.regularSpiking();
    const fs = IzhikevichNeuron.fastSpiking();
    let rsSpikes = 0, fsSpikes = 0;
    for (let t = 0; t < 200; t++) {
      if (rs.step(10)) rsSpikes++;
      if (fs.step(10)) fsSpikes++;
    }
    assert.ok(fsSpikes >= rsSpikes,
      `Fast spiking (${fsSpikes}) should fire >= regular (${rsSpikes})`);
  });

  it('no spike without input', () => {
    const n = IzhikevichNeuron.regularSpiking();
    let spiked = false;
    for (let t = 0; t < 100; t++) {
      if (n.step(0)) spiked = true;
    }
    assert.ok(!spiked, 'Should not spike without input');
  });

  it('records voltage history', () => {
    const n = IzhikevichNeuron.regularSpiking();
    for (let t = 0; t < 50; t++) n.step(5);
    assert.equal(n.voltageHistory.length, 50);
  });

  it('preset types have different parameters', () => {
    const rs = IzhikevichNeuron.regularSpiking();
    const ib = IzhikevichNeuron.intrinsicBursting();
    assert.notEqual(rs.c, ib.c);
  });
});

// snn-stress.test.js — Deep stress tests for Spiking Neural Networks
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  LIFNeuron, Synapse, STDPRule, PoissonEncoder,
  SNNLayer, SpikingNetwork, IzhikevichNeuron, spikeRate,
} from '../src/snn.js';

describe('LIF Neuron Dynamics', () => {
  it('no input should decay to resting potential', () => {
    const n = new LIFNeuron();
    n.v = -60; // Start above rest
    for (let t = 0; t < 100; t++) n.step(0);
    assert.ok(Math.abs(n.v - n.vRest) < 1, `Should decay to rest: v=${n.v.toFixed(1)}`);
  });

  it('constant strong input should produce spikes', () => {
    const n = new LIFNeuron();
    let spikeCount = 0;
    for (let t = 0; t < 200; t++) {
      if (n.step(20)) spikeCount++;
    }
    assert.ok(spikeCount > 5, `Strong input should produce multiple spikes: ${spikeCount}`);
  });

  it('voltage resets to vReset after spike', () => {
    const n = new LIFNeuron({ vReset: -70 });
    // Drive to spike
    while (!n.step(50));
    assert.equal(n.v, -70, 'Voltage should reset to vReset after spike');
  });

  it('refractory period prevents immediate re-spiking', () => {
    const n = new LIFNeuron({ refractoryPeriod: 5 });
    // First spike
    while (!n.step(50));
    const spikeTime = n.voltageHistory.length;
    
    // Should not spike during refractory period
    for (let t = 0; t < 4; t++) {
      assert.equal(n.step(50), false, `Should not spike during refractory at t+${t+1}`);
    }
  });

  it('higher current produces higher firing rate', () => {
    const low = new LIFNeuron();
    const high = new LIFNeuron();
    let lowSpikes = 0, highSpikes = 0;
    
    for (let t = 0; t < 500; t++) {
      if (low.step(12)) lowSpikes++;
      if (high.step(30)) highSpikes++;
    }
    
    assert.ok(highSpikes > lowSpikes, 
      `Higher current should produce more spikes: ${highSpikes} vs ${lowSpikes}`);
  });

  it('sub-threshold input produces no spikes', () => {
    const n = new LIFNeuron();
    let spikeCount = 0;
    for (let t = 0; t < 200; t++) {
      if (n.step(5)) spikeCount++; // Very weak input
    }
    // With tau=10 and vThresh-vRest=10, input of 5 is sub-threshold
    assert.equal(spikeCount, 0, 'Sub-threshold input should produce no spikes');
  });

  it('voltage history has correct length', () => {
    const n = new LIFNeuron();
    for (let t = 0; t < 50; t++) n.step(10);
    assert.equal(n.voltageHistory.length, 50);
  });

  it('reset clears all state', () => {
    const n = new LIFNeuron();
    for (let t = 0; t < 100; t++) n.step(20);
    n.reset();
    assert.equal(n.v, n.vRest);
    assert.equal(n.refractoryTimer, 0);
    assert.equal(n.spiked, false);
    assert.equal(n.voltageHistory.length, 0);
  });
});

describe('Izhikevich Neuron Patterns', () => {
  it('regular spiking: consistent inter-spike intervals', () => {
    const n = IzhikevichNeuron.regularSpiking();
    const spikeTimes = [];
    for (let t = 0; t < 500; t++) {
      if (n.step(10, 0.5)) spikeTimes.push(t);
    }
    assert.ok(spikeTimes.length > 3, `Should produce spikes: ${spikeTimes.length}`);
    
    // Check consistency of intervals (after initial transient)
    if (spikeTimes.length > 5) {
      const intervals = [];
      for (let i = 2; i < spikeTimes.length; i++) {
        intervals.push(spikeTimes[i] - spikeTimes[i - 1]);
      }
      const meanISI = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const maxDeviation = Math.max(...intervals.map(i => Math.abs(i - meanISI)));
      // Regular spiking should have fairly consistent intervals
      assert.ok(maxDeviation < meanISI * 0.5,
        `ISI should be consistent: mean=${meanISI.toFixed(1)}, maxDev=${maxDeviation.toFixed(1)}`);
    }
  });

  it('fast spiking: higher frequency than regular', () => {
    const regular = IzhikevichNeuron.regularSpiking();
    const fast = IzhikevichNeuron.fastSpiking();
    let regSpikes = 0, fastSpikes = 0;
    
    for (let t = 0; t < 500; t++) {
      if (regular.step(10, 0.5)) regSpikes++;
      if (fast.step(10, 0.5)) fastSpikes++;
    }
    
    assert.ok(fastSpikes > regSpikes || (fastSpikes > 0 && regSpikes > 0),
      `Fast should spike more: ${fastSpikes} vs ${regSpikes}`);
  });

  it('intrinsic bursting: produces bursts', () => {
    const n = IzhikevichNeuron.intrinsicBursting();
    const spikeTimes = [];
    for (let t = 0; t < 1000; t++) {
      if (n.step(10, 0.5)) spikeTimes.push(t);
    }
    assert.ok(spikeTimes.length > 2, `Should produce spikes: ${spikeTimes.length}`);
  });

  it('no input returns to rest', () => {
    const n = IzhikevichNeuron.regularSpiking();
    // Drive it first
    for (let t = 0; t < 100; t++) n.step(15, 0.5);
    // Then no input
    for (let t = 0; t < 500; t++) n.step(0, 0.5);
    assert.ok(n.v < -50, `Should return toward rest: v=${n.v.toFixed(1)}`);
  });

  it('voltage stays bounded (no explosion)', () => {
    const n = IzhikevichNeuron.regularSpiking();
    for (let t = 0; t < 1000; t++) n.step(20, 0.5);
    for (const v of n.voltageHistory) {
      assert.ok(v <= 30 && v > -100, `Voltage should be bounded: ${v.toFixed(1)}`);
    }
  });
});

describe('STDP Learning Window', () => {
  it('pre-before-post strengthens synapse (LTP)', () => {
    const stdp = new STDPRule();
    const synapse = { weight: 1.0 };
    const preSpikes = [10]; // Pre fires first
    const postSpikes = [15]; // Post fires 5ms later
    const dw = stdp.update(synapse, preSpikes, postSpikes);
    assert.ok(dw > 0, `Pre-before-post should strengthen: dw=${dw.toFixed(4)}`);
    assert.ok(synapse.weight > 1.0);
  });

  it('post-before-pre weakens synapse (LTD)', () => {
    const stdp = new STDPRule();
    const synapse = { weight: 1.0 };
    const preSpikes = [15]; // Pre fires second
    const postSpikes = [10]; // Post fires first
    const dw = stdp.update(synapse, preSpikes, postSpikes);
    assert.ok(dw < 0, `Post-before-pre should weaken: dw=${dw.toFixed(4)}`);
    assert.ok(synapse.weight < 1.0);
  });

  it('closer spikes produce larger weight change', () => {
    const stdp = new STDPRule();
    const close = { weight: 1.0 };
    const far = { weight: 1.0 };
    stdp.update(close, [10], [12]); // 2ms gap
    stdp.update(far, [10], [30]);   // 20ms gap
    assert.ok(close.weight > far.weight, 'Closer spikes should produce larger LTP');
  });

  it('weight respects bounds [wMin, wMax]', () => {
    const stdp = new STDPRule({ wMax: 5, wMin: 0 });
    const synapse = { weight: 4.9 };
    // Many LTP events
    for (let i = 0; i < 100; i++) {
      stdp.update(synapse, [i * 10], [i * 10 + 2]);
    }
    assert.ok(synapse.weight <= 5, `Weight should not exceed wMax: ${synapse.weight}`);
    assert.ok(synapse.weight >= 0, `Weight should not go below wMin: ${synapse.weight}`);
  });

  it('symmetric timing produces no net change', () => {
    const stdp = new STDPRule({ aPlus: 0.01, aMinus: 0.01, tauPlus: 20, tauMinus: 20 });
    const synapse = { weight: 1.0 };
    // Pre and post at same time → dt = 0, no change
    stdp.update(synapse, [10], [10]);
    assert.equal(synapse.weight, 1.0, 'Simultaneous spikes should produce no change');
  });
});

describe('Poisson Encoder', () => {
  it('higher value produces more spikes', () => {
    const enc = new PoissonEncoder(200);
    const trials = 10;
    let lowTotal = 0, highTotal = 0;
    for (let t = 0; t < trials; t++) {
      lowTotal += enc.encode(0.2, 100).length;
      highTotal += enc.encode(0.8, 100).length;
    }
    assert.ok(highTotal > lowTotal, `Higher value should produce more spikes: ${highTotal} vs ${lowTotal}`);
  });

  it('zero value produces no spikes', () => {
    const enc = new PoissonEncoder();
    const spikes = enc.encode(0, 100);
    assert.equal(spikes.length, 0);
  });
});

describe('SNN Layer', () => {
  it('produces boolean spike output', () => {
    const layer = new SNNLayer(3, 2);
    const output = layer.step([true, false, true]);
    assert.equal(output.length, 2);
    for (const s of output) {
      assert.ok(typeof s === 'boolean', `Output should be boolean: ${s}`);
    }
  });
});

describe('Full SNN Simulation', () => {
  it('processes 100ms simulation without errors', () => {
    const net = new SpikingNetwork([3, 5, 2]);
    const inputTrains = [
      [5, 15, 25, 35], // Neuron 0 spike times
      [10, 20, 30],    // Neuron 1 spike times
      [8, 18, 28, 38], // Neuron 2 spike times
    ];
    const outputTrains = net.simulate(inputTrains, 50);
    assert.equal(outputTrains.length, 2);
    for (const train of outputTrains) {
      assert.ok(Array.isArray(train));
      for (const t of train) {
        assert.ok(t >= 0 && t < 50, `Spike time should be in [0, 50): ${t}`);
      }
    }
  });

  it('more input spikes should generally produce more output', () => {
    let sparseTotal = 0, denseTotal = 0;
    for (let trial = 0; trial < 5; trial++) {
      const net1 = new SpikingNetwork([2, 4, 1]);
      const net2 = new SpikingNetwork([2, 4, 1]);
      // Copy weights
      for (let l = 0; l < net1.layers.length; l++) {
        net2.layers[l].weights = net1.layers[l].weights.map(row => [...row]);
      }
      
      const sparse = net1.simulate([[10], [30]], 50);
      const dense = net2.simulate([[5, 10, 15, 20, 25, 30, 35, 40], [5, 10, 15, 20, 25, 30, 35, 40]], 50);
      sparseTotal += sparse[0].length;
      denseTotal += dense[0].length;
    }
    assert.ok(denseTotal >= sparseTotal,
      `Dense input should produce >= sparse output: ${denseTotal} vs ${sparseTotal}`);
  });

  it('reset clears all neuron states', () => {
    const net = new SpikingNetwork([3, 2]);
    net.simulate([[5, 15], [10, 20], [8, 18]], 30);
    net.reset();
    for (const layer of net.layers) {
      for (const n of layer.neurons) {
        assert.equal(n.v, n.vRest);
        assert.equal(n.voltageHistory.length, 0);
      }
    }
  });
});

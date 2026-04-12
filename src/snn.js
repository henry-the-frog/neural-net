// snn.js — Spiking Neural Network
// Leaky Integrate-and-Fire (LIF) neurons with STDP learning

// ===== LIF Neuron =====
// dV/dt = -(V - V_rest) / tau + I(t) / C
// When V >= V_thresh: spike, reset to V_reset

export class LIFNeuron {
  constructor({
    vRest = -65,      // mV (resting potential)
    vThresh = -55,    // mV (spike threshold)
    vReset = -70,     // mV (reset after spike)
    tau = 10,         // ms (membrane time constant)
    refractoryPeriod = 2, // ms
  } = {}) {
    this.vRest = vRest;
    this.vThresh = vThresh;
    this.vReset = vReset;
    this.tau = tau;
    this.refractoryPeriod = refractoryPeriod;

    this.v = vRest;
    this.refractoryTimer = 0;
    this.spiked = false;
    this.spikeTimes = [];
    this.voltageHistory = [];
  }

  step(current, dt = 1) {
    this.spiked = false;

    if (this.refractoryTimer > 0) {
      this.refractoryTimer -= dt;
      this.voltageHistory.push(this.v);
      return false;
    }

    // Leaky integration
    const dv = (-(this.v - this.vRest) + current) / this.tau * dt;
    this.v += dv;
    this.voltageHistory.push(this.v);

    // Check for spike
    if (this.v >= this.vThresh) {
      this.spiked = true;
      this.v = this.vReset;
      this.refractoryTimer = this.refractoryPeriod;
      return true;
    }

    return false;
  }

  reset() {
    this.v = this.vRest;
    this.refractoryTimer = 0;
    this.spiked = false;
    this.spikeTimes = [];
    this.voltageHistory = [];
  }
}

// ===== Synapse =====
export class Synapse {
  constructor(preNeuron, postNeuron, weight = 1, delay = 1) {
    this.pre = preNeuron;
    this.post = postNeuron;
    this.weight = weight;
    this.delay = delay;
    this.spikeQueue = []; // Delayed spike delivery
  }

  // Queue a spike for delayed delivery
  propagate(time) {
    if (this.pre.spiked) {
      this.spikeQueue.push(time + this.delay);
    }
  }

  // Deliver current at this timestep
  deliver(time) {
    let current = 0;
    const newQueue = [];
    for (const t of this.spikeQueue) {
      if (Math.abs(t - time) < 0.5) {
        current += this.weight;
      } else if (t > time) {
        newQueue.push(t);
      }
    }
    this.spikeQueue = newQueue;
    return current;
  }
}

// ===== STDP (Spike-Timing-Dependent Plasticity) =====
export class STDPRule {
  constructor({
    aPlus = 0.01,    // LTP amplitude (pre before post → strengthen)
    aMinus = 0.012,  // LTD amplitude (post before pre → weaken)
    tauPlus = 20,    // ms (LTP time window)
    tauMinus = 20,   // ms (LTD time window)
    wMax = 5,        // Maximum weight
    wMin = 0,        // Minimum weight
  } = {}) {
    this.aPlus = aPlus;
    this.aMinus = aMinus;
    this.tauPlus = tauPlus;
    this.tauMinus = tauMinus;
    this.wMax = wMax;
    this.wMin = wMin;
  }

  // Compute weight change based on spike timing
  update(synapse, preSpikeTimes, postSpikeTimes) {
    let dw = 0;

    // For each pair of pre and post spikes
    for (const tPre of preSpikeTimes) {
      for (const tPost of postSpikeTimes) {
        const dt = tPost - tPre;
        if (dt > 0) {
          // Pre before post → LTP (potentiation)
          dw += this.aPlus * Math.exp(-dt / this.tauPlus);
        } else if (dt < 0) {
          // Post before pre → LTD (depression)
          dw -= this.aMinus * Math.exp(dt / this.tauMinus);
        }
      }
    }

    synapse.weight = Math.max(this.wMin, Math.min(this.wMax, synapse.weight + dw));
    return dw;
  }
}

// ===== Poisson Spike Generator =====
// Converts continuous value to spike train
export class PoissonEncoder {
  constructor(maxRate = 100) { // max firing rate in Hz
    this.maxRate = maxRate;
  }

  // Encode value [0, 1] as spike train over duration ms
  encode(value, duration, dt = 1) {
    const spikes = [];
    const rate = value * this.maxRate;
    for (let t = 0; t < duration; t += dt) {
      if (Math.random() < rate * dt / 1000) {
        spikes.push(t);
      }
    }
    return spikes;
  }
}

// ===== Rate Decoder =====
// Converts spike train to rate
export function spikeRate(spikeTimes, duration) {
  return spikeTimes.length / (duration / 1000); // Hz
}

// ===== SNN Layer =====
export class SNNLayer {
  constructor(inputSize, outputSize, neuronParams = {}) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;

    // Create neurons
    this.neurons = Array.from({ length: outputSize }, () => new LIFNeuron(neuronParams));

    // Create synapses (all-to-all)
    this.weights = [];
    for (let i = 0; i < inputSize; i++) {
      this.weights[i] = [];
      for (let j = 0; j < outputSize; j++) {
        this.weights[i][j] = (Math.random() - 0.2) * 2; // Mostly excitatory
      }
    }
  }

  // Simulate one timestep
  step(inputSpikes, dt = 1) {
    // inputSpikes: array of booleans (true = spike this timestep)
    const outputSpikes = new Array(this.outputSize).fill(false);

    for (let j = 0; j < this.outputSize; j++) {
      // Compute total input current
      let current = 0;
      for (let i = 0; i < this.inputSize; i++) {
        if (inputSpikes[i]) {
          current += this.weights[i][j];
        }
      }

      // Step the neuron
      outputSpikes[j] = this.neurons[j].step(current, dt);
    }

    return outputSpikes;
  }

  reset() {
    for (const n of this.neurons) n.reset();
  }
}

// ===== SNN Network =====
export class SpikingNetwork {
  constructor(layerSizes, neuronParams = {}) {
    this.layers = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      this.layers.push(new SNNLayer(layerSizes[l], layerSizes[l + 1], neuronParams));
    }
    this.stdp = new STDPRule();
  }

  // Simulate for given number of timesteps
  simulate(inputSpikeTrains, duration, dt = 1) {
    // inputSpikeTrains: [inputSize] arrays of spike times
    const outputSpikeTimes = this.layers[this.layers.length - 1].neurons.map(() => []);

    for (let t = 0; t < duration; t += dt) {
      // Convert input spike trains to boolean for this timestep
      let spikes = inputSpikeTrains.map(train =>
        train.some(st => Math.abs(st - t) < dt / 2)
      );

      // Propagate through layers
      for (const layer of this.layers) {
        spikes = layer.step(spikes, dt);
      }

      // Record output spikes
      const lastLayer = this.layers[this.layers.length - 1];
      for (let j = 0; j < lastLayer.outputSize; j++) {
        if (spikes[j]) outputSpikeTimes[j].push(t);
      }
    }

    return outputSpikeTimes;
  }

  // Apply STDP learning
  applySTDP() {
    for (const layer of this.layers) {
      for (let i = 0; i < layer.inputSize; i++) {
        for (let j = 0; j < layer.outputSize; j++) {
          // Get spike times (from voltage history / spike flags)
          // This is simplified — in practice you'd track spike times per neuron
          const preSpikes = []; // Would come from input layer
          const postSpikes = layer.neurons[j].spikeTimes;
          if (preSpikes.length > 0 && postSpikes.length > 0) {
            const synapse = { weight: layer.weights[i][j] };
            this.stdp.update(synapse, preSpikes, postSpikes);
            layer.weights[i][j] = synapse.weight;
          }
        }
      }
    }
  }

  reset() {
    for (const layer of this.layers) layer.reset();
  }
}

// ===== Izhikevich Neuron (more biologically realistic) =====
export class IzhikevichNeuron {
  constructor({
    a = 0.02,  // Recovery time scale
    b = 0.2,   // Sensitivity of recovery
    c = -65,   // Reset potential (mV)
    d = 8,     // Reset recovery increment
  } = {}) {
    this.a = a;
    this.b = b;
    this.c = c;
    this.d = d;
    this.v = -65;  // Membrane potential
    this.u = this.b * this.v; // Recovery variable
    this.spiked = false;
    this.voltageHistory = [];
  }

  step(current, dt = 1) {
    this.spiked = false;

    // Izhikevich model equations
    const dv = 0.04 * this.v * this.v + 5 * this.v + 140 - this.u + current;
    const du = this.a * (this.b * this.v - this.u);

    this.v += dv * dt;
    this.u += du * dt;

    if (this.v >= 30) {
      this.spiked = true;
      this.voltageHistory.push(30); // Record spike peak, not overshoot
      this.v = this.c;
      this.u += this.d;
      return true;
    }

    this.voltageHistory.push(this.v);

    return false;
  }

  reset() {
    this.v = -65;
    this.u = this.b * this.v;
    this.spiked = false;
    this.voltageHistory = [];
  }

  // Preset neuron types
  static regularSpiking() { return new IzhikevichNeuron({ a: 0.02, b: 0.2, c: -65, d: 8 }); }
  static intrinsicBursting() { return new IzhikevichNeuron({ a: 0.02, b: 0.2, c: -55, d: 4 }); }
  static chattering() { return new IzhikevichNeuron({ a: 0.02, b: 0.2, c: -50, d: 2 }); }
  static fastSpiking() { return new IzhikevichNeuron({ a: 0.1, b: 0.2, c: -65, d: 2 }); }
}

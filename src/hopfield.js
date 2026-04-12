// hopfield.js — Hopfield Network
// Associative memory with binary patterns, energy minimization

// ===== Classical Hopfield Network =====
// Binary states {-1, +1}, symmetric weights, no self-connections
// Energy: E = -0.5 * sum_ij(w_ij * s_i * s_j) - sum_i(theta_i * s_i)

export class HopfieldNetwork {
  constructor(size) {
    this.size = size;
    // Weight matrix (symmetric, zero diagonal)
    this.weights = Array.from({ length: size }, () => new Array(size).fill(0));
    this.thresholds = new Array(size).fill(0);
    this.state = new Array(size).fill(-1);
    this.storedPatterns = [];
  }

  // Store patterns using Hebbian learning (outer product rule)
  // patterns: array of arrays of {-1, +1}
  store(patterns) {
    // Reset weights
    for (let i = 0; i < this.size; i++)
      for (let j = 0; j < this.size; j++)
        this.weights[i][j] = 0;

    for (const pattern of patterns) {
      if (pattern.length !== this.size) throw new Error(`Pattern size mismatch: ${pattern.length} != ${this.size}`);
      this.storedPatterns.push([...pattern]);

      for (let i = 0; i < this.size; i++) {
        for (let j = 0; j < this.size; j++) {
          if (i !== j) {
            this.weights[i][j] += pattern[i] * pattern[j] / this.size;
          }
        }
      }
    }
  }

  // Compute energy of current state
  energy() {
    let E = 0;
    for (let i = 0; i < this.size; i++) {
      for (let j = i + 1; j < this.size; j++) {
        E -= this.weights[i][j] * this.state[i] * this.state[j];
      }
      E -= this.thresholds[i] * this.state[i];
    }
    return E;
  }

  // Asynchronous update: pick random neuron, update based on local field
  stepAsync(neuronIdx = null) {
    const i = neuronIdx !== null ? neuronIdx : Math.floor(Math.random() * this.size);
    let h = this.thresholds[i];
    for (let j = 0; j < this.size; j++) {
      h += this.weights[i][j] * this.state[j];
    }
    this.state[i] = h >= 0 ? 1 : -1;
    return i;
  }

  // Synchronous update: update all neurons simultaneously
  stepSync() {
    const newState = new Array(this.size);
    for (let i = 0; i < this.size; i++) {
      let h = this.thresholds[i];
      for (let j = 0; j < this.size; j++) {
        h += this.weights[i][j] * this.state[j];
      }
      newState[i] = h >= 0 ? 1 : -1;
    }
    this.state = newState;
  }

  // Recall: set initial state and iterate until convergence
  recall(probe, maxIter = 100, mode = 'async') {
    this.state = [...probe];
    const energyHistory = [this.energy()];

    for (let iter = 0; iter < maxIter; iter++) {
      const prevState = [...this.state];

      if (mode === 'async') {
        // One full sweep (each neuron updated once in random order)
        const order = shuffleArray(Array.from({ length: this.size }, (_, i) => i));
        for (const i of order) this.stepAsync(i);
      } else {
        this.stepSync();
      }

      energyHistory.push(this.energy());

      // Check convergence
      if (this.state.every((s, i) => s === prevState[i])) {
        return { state: [...this.state], converged: true, iterations: iter + 1, energyHistory };
      }
    }

    return { state: [...this.state], converged: false, iterations: maxIter, energyHistory };
  }

  // Check overlap (similarity) with stored patterns
  overlap(pattern) {
    let sum = 0;
    for (let i = 0; i < this.size; i++) {
      sum += this.state[i] * pattern[i];
    }
    return sum / this.size; // Returns value in [-1, 1]
  }

  // Find closest stored pattern
  closestPattern() {
    let bestOverlap = -Infinity;
    let bestIdx = -1;
    for (let p = 0; p < this.storedPatterns.length; p++) {
      const ov = this.overlap(this.storedPatterns[p]);
      if (ov > bestOverlap) {
        bestOverlap = ov;
        bestIdx = p;
      }
    }
    return { index: bestIdx, overlap: bestOverlap };
  }

  // Theoretical capacity: ~0.138 * N patterns
  theoreticalCapacity() {
    return Math.floor(0.138 * this.size);
  }
}

// ===== Modern Hopfield Network =====
// Continuous states, exponential energy, higher capacity
// Energy: E = -log(sum_mu exp(beta * pattern_mu . state))
export class ModernHopfieldNetwork {
  constructor(size, beta = 1) {
    this.size = size;
    this.beta = beta;
    this.patterns = []; // Stored patterns (can be continuous)
  }

  store(patterns) {
    this.patterns = patterns.map(p => [...p]);
  }

  // Softmax attention over stored patterns
  retrieve(query, iterations = 10) {
    let state = [...query];

    for (let iter = 0; iter < iterations; iter++) {
      // Compute similarities
      const logits = this.patterns.map(p => {
        let dot = 0;
        for (let i = 0; i < this.size; i++) dot += p[i] * state[i];
        return this.beta * dot;
      });

      // Softmax
      const maxLogit = Math.max(...logits);
      const exps = logits.map(l => Math.exp(l - maxLogit));
      const sumExp = exps.reduce((a, b) => a + b, 0);
      const weights = exps.map(e => e / sumExp);

      // Weighted sum of patterns
      const newState = new Array(this.size).fill(0);
      for (let mu = 0; mu < this.patterns.length; mu++) {
        for (let i = 0; i < this.size; i++) {
          newState[i] += weights[mu] * this.patterns[mu][i];
        }
      }

      state = newState;
    }

    return state;
  }

  // Energy of a state
  energy(state) {
    const logits = this.patterns.map(p => {
      let dot = 0;
      for (let i = 0; i < this.size; i++) dot += p[i] * state[i];
      return this.beta * dot;
    });
    const maxLogit = Math.max(...logits);
    const lse = maxLogit + Math.log(logits.map(l => Math.exp(l - maxLogit)).reduce((a, b) => a + b, 0));
    return -lse;
  }
}

// ===== Boltzmann Machine =====
// Stochastic Hopfield network with temperature
export class BoltzmannMachine {
  constructor(size, temperature = 1) {
    this.size = size;
    this.temperature = temperature;
    this.weights = Array.from({ length: size }, () => new Array(size).fill(0));
    this.biases = new Array(size).fill(0);
    this.state = Array.from({ length: size }, () => Math.random() > 0.5 ? 1 : -1);
  }

  // Stochastic update with simulated annealing
  step() {
    const i = Math.floor(Math.random() * this.size);
    let h = this.biases[i];
    for (let j = 0; j < this.size; j++) {
      h += this.weights[i][j] * this.state[j];
    }
    // Sigmoid probability
    const prob = 1 / (1 + Math.exp(-2 * h / this.temperature));
    this.state[i] = Math.random() < prob ? 1 : -1;
    return i;
  }

  energy() {
    let E = 0;
    for (let i = 0; i < this.size; i++) {
      for (let j = i + 1; j < this.size; j++) {
        E -= this.weights[i][j] * this.state[i] * this.state[j];
      }
      E -= this.biases[i] * this.state[i];
    }
    return E;
  }

  // Run for given steps, optionally with annealing
  run(steps, anneal = false) {
    const energies = [];
    for (let s = 0; s < steps; s++) {
      if (anneal) {
        this.temperature = Math.max(0.01, 1 - s / steps);
      }
      this.step();
      if (s % 10 === 0) energies.push(this.energy());
    }
    return energies;
  }
}

// ===== Utility =====
function shuffleArray(arr) {
  const shuffled = [...arr];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// Generate random binary pattern
export function randomPattern(size) {
  return Array.from({ length: size }, () => Math.random() > 0.5 ? 1 : -1);
}

// Corrupt pattern by flipping bits
export function corruptPattern(pattern, flipRate = 0.2) {
  return pattern.map(v => Math.random() < flipRate ? -v : v);
}

// Hamming distance between two binary patterns
export function hammingDistance(a, b) {
  let dist = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) dist++;
  }
  return dist;
}

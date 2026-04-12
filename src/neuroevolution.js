// neuroevolution.js — Genetic Algorithm for Neural Network Optimization
// Evolve weights without backpropagation, NEAT-inspired topology mutation

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';

// ===== Simple Feedforward Genome =====
// Genome = flat array of all weights + biases
export class NeuralGenome {
  constructor(layerSizes) {
    this.layerSizes = layerSizes;
    this.layers = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      this.layers.push(new Dense(layerSizes[l], layerSizes[l + 1], l < layerSizes.length - 2 ? 'tanh' : 'linear'));
    }
    this.fitness = -Infinity;
  }

  // Forward pass
  forward(input) {
    let x = input;
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    return x;
  }

  // Get flat weight vector
  getWeights() {
    const weights = [];
    for (const layer of this.layers) {
      for (let i = 0; i < layer.weights.rows; i++)
        for (let j = 0; j < layer.weights.cols; j++)
          weights.push(layer.weights.get(i, j));
      for (let j = 0; j < layer.biases.cols; j++)
        weights.push(layer.biases.get(0, j));
    }
    return weights;
  }

  // Set weights from flat vector
  setWeights(weights) {
    let idx = 0;
    for (const layer of this.layers) {
      for (let i = 0; i < layer.weights.rows; i++)
        for (let j = 0; j < layer.weights.cols; j++)
          layer.weights.set(i, j, weights[idx++]);
      for (let j = 0; j < layer.biases.cols; j++)
        layer.biases.set(0, j, weights[idx++]);
    }
  }

  // Clone this genome
  clone() {
    const g = new NeuralGenome(this.layerSizes);
    g.setWeights(this.getWeights());
    g.fitness = this.fitness;
    return g;
  }

  paramCount() {
    return this.getWeights().length;
  }
}

// ===== Selection Methods =====

export function tournamentSelection(population, tournamentSize = 3) {
  const candidates = [];
  for (let i = 0; i < tournamentSize; i++) {
    candidates.push(population[Math.floor(Math.random() * population.length)]);
  }
  candidates.sort((a, b) => b.fitness - a.fitness);
  return candidates[0];
}

export function rouletteSelection(population) {
  const minFitness = Math.min(...population.map(g => g.fitness));
  const shifted = population.map(g => g.fitness - minFitness + 1e-6);
  const total = shifted.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let i = 0; i < population.length; i++) {
    r -= shifted[i];
    if (r <= 0) return population[i];
  }
  return population[population.length - 1];
}

export function rankSelection(population) {
  const sorted = [...population].sort((a, b) => a.fitness - b.fitness);
  const n = sorted.length;
  const totalRank = n * (n + 1) / 2;
  let r = Math.random() * totalRank;
  for (let i = 0; i < n; i++) {
    r -= (i + 1);
    if (r <= 0) return sorted[i];
  }
  return sorted[n - 1];
}

// ===== Crossover Methods =====

export function uniformCrossover(parent1, parent2) {
  const w1 = parent1.getWeights();
  const w2 = parent2.getWeights();
  const childWeights = w1.map((v, i) => Math.random() < 0.5 ? v : w2[i]);
  const child = parent1.clone();
  child.setWeights(childWeights);
  child.fitness = -Infinity;
  return child;
}

export function singlePointCrossover(parent1, parent2) {
  const w1 = parent1.getWeights();
  const w2 = parent2.getWeights();
  const point = Math.floor(Math.random() * w1.length);
  const childWeights = [...w1.slice(0, point), ...w2.slice(point)];
  const child = parent1.clone();
  child.setWeights(childWeights);
  child.fitness = -Infinity;
  return child;
}

export function blendCrossover(parent1, parent2, alpha = 0.5) {
  const w1 = parent1.getWeights();
  const w2 = parent2.getWeights();
  const childWeights = w1.map((v, i) => {
    const t = Math.random() * (1 + 2 * alpha) - alpha;
    return v * (1 - t) + w2[i] * t;
  });
  const child = parent1.clone();
  child.setWeights(childWeights);
  child.fitness = -Infinity;
  return child;
}

// ===== Mutation Methods =====

export function gaussianMutation(genome, mutationRate = 0.1, sigma = 0.1) {
  const weights = genome.getWeights();
  const mutated = weights.map(w =>
    Math.random() < mutationRate ? w + gaussianRandom() * sigma : w
  );
  genome.setWeights(mutated);
}

export function uniformMutation(genome, mutationRate = 0.05, range = 0.5) {
  const weights = genome.getWeights();
  const mutated = weights.map(w =>
    Math.random() < mutationRate ? w + (Math.random() * 2 - 1) * range : w
  );
  genome.setWeights(mutated);
}

// ===== Genetic Algorithm =====
export class GeneticAlgorithm {
  constructor(layerSizes, {
    populationSize = 50,
    eliteCount = 5,
    mutationRate = 0.1,
    mutationSigma = 0.1,
    crossoverRate = 0.7,
    selectionMethod = 'tournament',
    crossoverMethod = 'uniform',
  } = {}) {
    this.layerSizes = layerSizes;
    this.populationSize = populationSize;
    this.eliteCount = eliteCount;
    this.mutationRate = mutationRate;
    this.mutationSigma = mutationSigma;
    this.crossoverRate = crossoverRate;
    this.selectionMethod = selectionMethod;
    this.crossoverMethod = crossoverMethod;

    // Initialize population
    this.population = Array.from({ length: populationSize }, () => new NeuralGenome(layerSizes));
    this.generation = 0;
    this.bestFitness = -Infinity;
    this.bestGenome = null;
    this.fitnessHistory = [];
  }

  // Evaluate fitness for all genomes
  evaluate(fitnessFn) {
    for (const genome of this.population) {
      genome.fitness = fitnessFn(genome);
    }
    this.population.sort((a, b) => b.fitness - a.fitness);

    if (this.population[0].fitness > this.bestFitness) {
      this.bestFitness = this.population[0].fitness;
      this.bestGenome = this.population[0].clone();
    }
    this.fitnessHistory.push(this.bestFitness);
  }

  // Create next generation
  evolve() {
    const newPopulation = [];

    // Elitism: keep best
    for (let i = 0; i < this.eliteCount && i < this.population.length; i++) {
      newPopulation.push(this.population[i].clone());
    }

    // Fill rest with offspring
    while (newPopulation.length < this.populationSize) {
      const select = this.selectionMethod === 'tournament' ? tournamentSelection :
                     this.selectionMethod === 'roulette' ? rouletteSelection : rankSelection;

      const parent1 = select(this.population);
      const parent2 = select(this.population);

      let child;
      if (Math.random() < this.crossoverRate) {
        const crossover = this.crossoverMethod === 'uniform' ? uniformCrossover :
                          this.crossoverMethod === 'single' ? singlePointCrossover : blendCrossover;
        child = crossover(parent1, parent2);
      } else {
        child = parent1.clone();
      }

      gaussianMutation(child, this.mutationRate, this.mutationSigma);
      child.fitness = -Infinity;
      newPopulation.push(child);
    }

    this.population = newPopulation;
    this.generation++;
  }

  // Run for N generations
  run(generations, fitnessFn) {
    for (let g = 0; g < generations; g++) {
      this.evaluate(fitnessFn);
      this.evolve();
    }
    // Final evaluation
    this.evaluate(fitnessFn);
    return this.bestGenome;
  }

  stats() {
    return {
      generation: this.generation,
      bestFitness: this.bestFitness,
      avgFitness: this.population.reduce((s, g) => s + g.fitness, 0) / this.population.length,
      worstFitness: this.population[this.population.length - 1].fitness,
    };
  }
}

// ===== Evolution Strategy (ES) =====
// Simpler alternative: perturb → evaluate → weighted update
export class EvolutionStrategy {
  constructor(layerSizes, {
    populationSize = 50,
    sigma = 0.1,
    learningRate = 0.01,
  } = {}) {
    this.genome = new NeuralGenome(layerSizes);
    this.populationSize = populationSize;
    this.sigma = sigma;
    this.learningRate = learningRate;
    this.fitnessHistory = [];
  }

  step(fitnessFn) {
    const baseWeights = this.genome.getWeights();
    const perturbations = [];
    const fitnesses = [];

    // Generate perturbations and evaluate
    for (let i = 0; i < this.populationSize; i++) {
      const noise = baseWeights.map(() => gaussianRandom());
      perturbations.push(noise);

      const candidate = this.genome.clone();
      candidate.setWeights(baseWeights.map((w, j) => w + this.sigma * noise[j]));
      fitnesses.push(fitnessFn(candidate));
    }

    // Normalize fitnesses
    const mean = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    const std = Math.sqrt(fitnesses.reduce((s, f) => s + (f - mean) ** 2, 0) / fitnesses.length) + 1e-8;
    const normalized = fitnesses.map(f => (f - mean) / std);

    // Weighted update
    const update = baseWeights.map((_, j) => {
      let sum = 0;
      for (let i = 0; i < this.populationSize; i++) {
        sum += normalized[i] * perturbations[i][j];
      }
      return sum / (this.populationSize * this.sigma);
    });

    const newWeights = baseWeights.map((w, j) => w + this.learningRate * update[j]);
    this.genome.setWeights(newWeights);
    this.genome.fitness = fitnessFn(this.genome);
    this.fitnessHistory.push(this.genome.fitness);

    return this.genome.fitness;
  }

  run(generations, fitnessFn) {
    for (let g = 0; g < generations; g++) {
      this.step(fitnessFn);
    }
    return this.genome;
  }
}

// ===== Utility =====
function gaussianRandom() {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

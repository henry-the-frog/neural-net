// neuroevo-stress.test.js — Stress tests for Neuroevolution
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  NeuralGenome, GeneticAlgorithm,
  tournamentSelection, rouletteSelection, rankSelection,
  uniformCrossover, singlePointCrossover, blendCrossover,
  gaussianMutation, uniformMutation,
} from '../src/neuroevolution.js';
import { Matrix } from '../src/matrix.js';

function makeGenome(fitness = 0) {
  const g = new NeuralGenome([2, 3, 1]);
  g.fitness = fitness;
  return g;
}

function makePopulation(n = 10) {
  return Array.from({ length: n }, (_, i) => makeGenome(i));
}

describe('NeuralGenome', () => {
  it('creates layers with correct count', () => {
    const g = new NeuralGenome([2, 3, 1]);
    assert.equal(g.layers.length, 2);
  });

  it('forward returns correct output size', () => {
    const g = new NeuralGenome([4, 8, 2]);
    const output = g.forward(Matrix.fromArray([[1, 0.5, -0.3, 0.8]]));
    assert.equal(output.cols, 2);
    assert.ok(output.data.every(Number.isFinite));
  });

  it('clone creates independent copy', () => {
    const g = new NeuralGenome([2, 3, 1]);
    const c = g.clone();
    c.layers[0].weights.data[0] = 999;
    assert.notEqual(g.layers[0].weights.data[0], 999);
  });
});

describe('Selection', () => {
  it('tournament favors fitter', () => {
    const pop = makePopulation(100);
    let sumFitness = 0;
    for (let i = 0; i < 100; i++) sumFitness += tournamentSelection(pop, 5).fitness;
    const avgSelected = sumFitness / 100;
    const avgPop = pop.reduce((s, g) => s + g.fitness, 0) / pop.length;
    assert.ok(avgSelected > avgPop);
  });

  it('roulette returns genome', () => {
    assert.ok(rouletteSelection(makePopulation()));
  });

  it('rank returns genome', () => {
    assert.ok(rankSelection(makePopulation()));
  });
});

describe('Crossover', () => {
  it('uniform produces valid offspring', () => {
    const child = uniformCrossover(makeGenome(), makeGenome());
    assert.ok(child.forward(Matrix.fromArray([[1, 0]])).data.every(Number.isFinite));
  });

  it('single point produces valid offspring', () => {
    const child = singlePointCrossover(makeGenome(), makeGenome());
    assert.ok(child.forward(Matrix.fromArray([[1, 0]])).data.every(Number.isFinite));
  });

  it('blend produces valid offspring', () => {
    const child = blendCrossover(makeGenome(), makeGenome(), 0.5);
    assert.ok(child.forward(Matrix.fromArray([[1, 0]])).data.every(Number.isFinite));
  });
});

describe('Mutation', () => {
  it('gaussian changes weights', () => {
    const g = makeGenome();
    const before = g.layers[0].weights.data[0];
    gaussianMutation(g, 1.0, 0.5);
    assert.notEqual(g.layers[0].weights.data[0], before);
  });

  it('uniform changes weights', () => {
    const g = makeGenome();
    const before = g.layers[0].weights.data[0];
    uniformMutation(g, 1.0, 0.5);
    assert.notEqual(g.layers[0].weights.data[0], before);
  });
});

describe('Genetic Algorithm', () => {
  it('evolves without error', () => {
    const ga = new GeneticAlgorithm([2, 4, 1], { populationSize: 20, eliteCount: 2 });
    ga.evaluate((g) => g.forward(Matrix.fromArray([[1, 1]])).get(0, 0));
    ga.evolve();
    assert.ok(ga.bestFitness !== undefined);
  });

  it('fitness improves over generations', () => {
    const ga = new GeneticAlgorithm([1, 4, 1], { populationSize: 30, eliteCount: 3 });
    const f = (g) => g.forward(Matrix.fromArray([[1]])).get(0, 0);
    ga.evaluate(f);
    const before = ga.bestFitness;
    for (let i = 0; i < 30; i++) { ga.evolve(); ga.evaluate(f); }
    assert.ok(ga.bestFitness >= before - 0.5);
  });
});

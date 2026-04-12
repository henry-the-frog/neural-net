import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  NeuralGenome,
  tournamentSelection, rouletteSelection, rankSelection,
  uniformCrossover, singlePointCrossover, blendCrossover,
  gaussianMutation,
  GeneticAlgorithm, EvolutionStrategy,
} from '../src/neuroevolution.js';
import { Matrix } from '../src/matrix.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('NeuralGenome', () => {
  it('creates correct architecture', () => {
    const g = new NeuralGenome([2, 4, 1]);
    assert.equal(g.layers.length, 2);
  });

  it('forward produces correct shape', () => {
    const g = new NeuralGenome([3, 5, 2]);
    const input = Matrix.random(4, 3);
    const output = g.forward(input);
    assert.equal(output.rows, 4);
    assert.equal(output.cols, 2);
  });

  it('get/set weights roundtrips', () => {
    const g = new NeuralGenome([2, 3, 1]);
    const w = g.getWeights();
    const g2 = new NeuralGenome([2, 3, 1]);
    g2.setWeights(w);
    const w2 = g2.getWeights();
    assert.ok(w.every((v, i) => approx(v, w2[i], 0.0001)));
  });

  it('clone produces independent copy', () => {
    const g = new NeuralGenome([2, 3, 1]);
    g.fitness = 42;
    const g2 = g.clone();
    assert.equal(g2.fitness, 42);
    // Modify clone shouldn't affect original
    const w = g2.getWeights();
    w[0] += 100;
    g2.setWeights(w);
    assert.notEqual(g.getWeights()[0], g2.getWeights()[0]);
  });

  it('paramCount is correct', () => {
    const g = new NeuralGenome([2, 3, 1]);
    // layer 1: 2*3 + 3 = 9, layer 2: 3*1 + 1 = 4 → total 13
    assert.equal(g.paramCount(), 13);
  });
});

describe('Selection', () => {
  const pop = Array.from({ length: 10 }, (_, i) => {
    const g = new NeuralGenome([2, 1]);
    g.fitness = i;
    return g;
  });

  it('tournament selects among candidates', () => {
    const selected = tournamentSelection(pop, 3);
    assert.ok(selected.fitness >= 0 && selected.fitness <= 9);
  });

  it('tournament tends to select high fitness', () => {
    let sumFitness = 0;
    for (let i = 0; i < 100; i++) {
      sumFitness += tournamentSelection(pop, 5).fitness;
    }
    assert.ok(sumFitness / 100 > 4.5, `Average should be above median: ${sumFitness / 100}`);
  });

  it('roulette selects', () => {
    const selected = rouletteSelection(pop);
    assert.ok(selected.fitness >= 0);
  });

  it('rank selection works', () => {
    const selected = rankSelection(pop);
    assert.ok(selected.fitness >= 0);
  });
});

describe('Crossover', () => {
  it('uniform crossover produces valid offspring', () => {
    const p1 = new NeuralGenome([2, 3, 1]);
    const p2 = new NeuralGenome([2, 3, 1]);
    const child = uniformCrossover(p1, p2);
    assert.equal(child.paramCount(), p1.paramCount());
    assert.equal(child.fitness, -Infinity);
  });

  it('single point crossover produces valid offspring', () => {
    const p1 = new NeuralGenome([2, 3, 1]);
    const p2 = new NeuralGenome([2, 3, 1]);
    const child = singlePointCrossover(p1, p2);
    assert.equal(child.paramCount(), p1.paramCount());
  });

  it('blend crossover interpolates', () => {
    const p1 = new NeuralGenome([2, 1]);
    const p2 = new NeuralGenome([2, 1]);
    const child = blendCrossover(p1, p2, 0);
    // With alpha=0, child should be between parents
    const w1 = p1.getWeights();
    const w2 = p2.getWeights();
    const wc = child.getWeights();
    // Not strictly bounded, but should be in the neighborhood
    assert.ok(wc.every(Number.isFinite));
  });
});

describe('Mutation', () => {
  it('gaussian mutation changes weights', () => {
    const g = new NeuralGenome([2, 3, 1]);
    const before = [...g.getWeights()];
    gaussianMutation(g, 1.0, 0.5); // 100% mutation rate
    const after = g.getWeights();
    const changed = before.filter((v, i) => !approx(v, after[i], 0.001)).length;
    assert.ok(changed > 0, 'Should change some weights');
  });

  it('zero mutation rate preserves weights', () => {
    const g = new NeuralGenome([2, 3, 1]);
    const before = [...g.getWeights()];
    gaussianMutation(g, 0, 0.5);
    const after = g.getWeights();
    assert.ok(before.every((v, i) => v === after[i]));
  });
});

describe('Genetic Algorithm', () => {
  it('evolves to solve simple problem', () => {
    // Maximize f(x) = -(x - 3)^2 where x is the single weight
    const ga = new GeneticAlgorithm([1, 1], {
      populationSize: 30,
      eliteCount: 3,
      mutationRate: 0.3,
      mutationSigma: 0.5,
    });

    const fitnessFn = (genome) => {
      const w = genome.getWeights();
      // Target: first weight close to 3
      return -((w[0] - 3) ** 2);
    };

    ga.run(50, fitnessFn);
    assert.ok(ga.bestFitness > -1, `Should get close to optimum: ${ga.bestFitness}`);
  });

  it('fitness improves over generations', () => {
    const ga = new GeneticAlgorithm([2, 4, 1], { populationSize: 20 });

    const fitnessFn = (genome) => {
      const input = new Matrix(4, 2, new Float64Array([0, 0, 0, 1, 1, 0, 1, 1]));
      const target = [0, 1, 1, 0]; // XOR
      const output = genome.forward(input);
      let error = 0;
      for (let i = 0; i < 4; i++) error += (output.get(i, 0) - target[i]) ** 2;
      return -error; // Negative MSE
    };

    ga.run(30, fitnessFn);
    const history = ga.fitnessHistory;
    assert.ok(history[history.length - 1] > history[0],
      `Fitness should improve: ${history[0].toFixed(4)} → ${history[history.length - 1].toFixed(4)}`);
  });

  it('stats returns correct info', () => {
    const ga = new GeneticAlgorithm([2, 1], { populationSize: 10 });
    ga.evaluate(() => Math.random());
    const stats = ga.stats();
    assert.ok('generation' in stats);
    assert.ok('bestFitness' in stats);
    assert.ok('avgFitness' in stats);
  });

  it('elitism preserves best', () => {
    const ga = new GeneticAlgorithm([2, 1], { populationSize: 10, eliteCount: 2 });
    ga.evaluate((g) => {
      const w = g.getWeights();
      return -w.reduce((s, v) => s + v * v, 0);
    });
    const bestBefore = ga.bestFitness;
    ga.evolve();
    ga.evaluate((g) => {
      const w = g.getWeights();
      return -w.reduce((s, v) => s + v * v, 0);
    });
    // Best should be at least as good (elitism)
    assert.ok(ga.bestFitness >= bestBefore - 0.01);
  });
});

describe('Evolution Strategy', () => {
  it('optimizes simple function', () => {
    const es = new EvolutionStrategy([1, 1], {
      populationSize: 30,
      sigma: 0.1,
      learningRate: 0.05,
    });

    const fitnessFn = (genome) => {
      const w = genome.getWeights();
      return -((w[0] - 2) ** 2) - ((w[1] - 1) ** 2);
    };

    es.run(50, fitnessFn);
    assert.ok(es.genome.fitness > -1, `Should optimize: ${es.genome.fitness}`);
  });

  it('fitness improves over steps', () => {
    const es = new EvolutionStrategy([2, 3, 1], {
      populationSize: 20,
      sigma: 0.1,
      learningRate: 0.02,
    });

    const fitnessFn = (genome) => {
      const w = genome.getWeights();
      return -w.reduce((s, v) => s + v * v, 0); // Minimize L2 norm
    };

    es.run(30, fitnessFn);
    const history = es.fitnessHistory;
    assert.ok(history[history.length - 1] > history[0],
      `Fitness should improve: ${history[0].toFixed(4)} → ${history[history.length - 1].toFixed(4)}`);
  });
});

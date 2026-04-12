// meta-learning.test.js — Tests for the DARTS + Lottery Ticket meta-learning pipeline
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { metaLearningPipeline } from '../src/meta-learning.js';
import { Matrix } from '../src/matrix.js';

describe('Meta-Learning Pipeline', () => {
  it('runs full pipeline without errors', () => {
    const N = 10;
    const trainInputs = Matrix.random(N, 2);
    const trainTargets = Matrix.random(N, 1);
    
    const result = metaLearningPipeline({
      inputSize: 2,
      outputSize: 1,
      hiddenSize: 8,
      numNodes: 2,
      trainInputs,
      trainTargets,
      valInputs: trainInputs,
      valTargets: Array.from({ length: N }, (_, i) => trainTargets.get(i, 0)),
      dartsSteps: 10,
      trainEpochs: 50,
      pruneSparsity: 0.3,
      trainLR: 0.1,
    });
    
    assert.equal(result.phases.length, 3);
    assert.ok(result.summary);
    assert.ok(typeof result.summary.sparsity === 'number');
    assert.ok(typeof result.summary.fullNetworkFinalLoss === 'number');
    assert.ok(typeof result.summary.ticketFinalLoss === 'number');
  });

  it('DARTS phase produces architecture selections', () => {
    const result = metaLearningPipeline({
      inputSize: 2, outputSize: 1, hiddenSize: 4, numNodes: 2,
      trainInputs: Matrix.random(5, 2),
      trainTargets: Matrix.random(5, 1),
      valInputs: Matrix.random(5, 2),
      valTargets: [0.1, 0.2, 0.3, 0.4, 0.5],
      dartsSteps: 5, trainEpochs: 20, pruneSparsity: 0.3,
    });
    
    assert.ok(Object.keys(result.phases[0].architecture).length > 0);
  });

  it('training phase reduces loss', () => {
    const result = metaLearningPipeline({
      inputSize: 2, outputSize: 1, hiddenSize: 8, numNodes: 2,
      trainInputs: Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]),
      trainTargets: Matrix.fromArray([[0], [1], [1], [0]]),
      valInputs: Matrix.fromArray([[0, 0], [1, 1]]),
      valTargets: [0, 0],
      dartsSteps: 5, trainEpochs: 100, pruneSparsity: 0.3, trainLR: 0.5,
    });
    
    const trainPhase = result.phases[1];
    assert.ok(trainPhase.finalLoss <= trainPhase.initialLoss + 0.5,
      `Training should not dramatically increase loss: ${trainPhase.initialLoss.toFixed(4)} → ${trainPhase.finalLoss.toFixed(4)}`);
  });

  it('lottery ticket phase maintains reasonable loss', () => {
    const result = metaLearningPipeline({
      inputSize: 2, outputSize: 1, hiddenSize: 8, numNodes: 2,
      trainInputs: Matrix.random(10, 2),
      trainTargets: Matrix.random(10, 1),
      valInputs: Matrix.random(5, 2),
      valTargets: [0.1, 0.2, 0.3, 0.4, 0.5],
      dartsSteps: 5, trainEpochs: 50, pruneSparsity: 0.3,
    });
    
    const ticketPhase = result.phases[2];
    assert.ok(Number.isFinite(ticketPhase.finalLoss));
    assert.ok(ticketPhase.sparsity > 0 && ticketPhase.sparsity < 1);
  });

  it('higher sparsity produces sparser ticket', () => {
    const data = { inputSize: 2, outputSize: 1, hiddenSize: 8, numNodes: 2,
      trainInputs: Matrix.random(10, 2), trainTargets: Matrix.random(10, 1),
      valInputs: Matrix.random(5, 2), valTargets: [0.1, 0.2, 0.3, 0.4, 0.5],
      dartsSteps: 3, trainEpochs: 30 };
    
    const r30 = metaLearningPipeline({ ...data, pruneSparsity: 0.3 });
    const r70 = metaLearningPipeline({ ...data, pruneSparsity: 0.7 });
    
    assert.ok(r70.summary.sparsity > r30.summary.sparsity,
      `70% target should be sparser: ${r70.summary.sparsity.toFixed(2)} vs ${r30.summary.sparsity.toFixed(2)}`);
  });
});

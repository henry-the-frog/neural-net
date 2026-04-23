// reward-model.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { RewardModel } from './reward-model.js';

describe('Reward Model', () => {
  test('forward produces scalar', () => {
    const rm = new RewardModel(8, 16);
    const encoding = new Float64Array(8).fill(0.5);
    const reward = rm.forward(encoding);
    assert.ok(typeof reward === 'number');
    assert.ok(isFinite(reward));
  });

  test('different inputs produce different rewards', () => {
    const rm = new RewardModel(8, 16);
    const r1 = rm.forward(new Float64Array([1, 0, 0, 0, 0, 0, 0, 0]));
    const r2 = rm.forward(new Float64Array([0, 0, 0, 0, 0, 0, 0, 1]));
    assert.notEqual(r1, r2);
  });

  test('preference loss is finite', () => {
    const rm = new RewardModel(4, 8);
    const pairs = [
      { chosen: new Float64Array([1, 1, 0, 0]), rejected: new Float64Array([0, 0, 1, 1]) },
    ];
    const { loss, accuracy } = rm.preferenceLoss(pairs);
    assert.ok(isFinite(loss));
    assert.ok(accuracy >= 0 && accuracy <= 1);
  });

  test('training improves accuracy', () => {
    const rm = new RewardModel(4, 16);
    const pairs = [];
    // Generate consistent preference data: higher first component → preferred
    for (let i = 0; i < 20; i++) {
      const chosen = new Float64Array([1 + Math.random(), Math.random(), Math.random(), Math.random()]);
      const rejected = new Float64Array([-1 + Math.random(), Math.random(), Math.random(), Math.random()]);
      pairs.push({ chosen, rejected });
    }
    
    const before = rm.preferenceLoss(pairs);
    
    // Train for several steps
    for (let step = 0; step < 50; step++) {
      rm.trainStep(pairs, 0.01);
    }
    
    const after = rm.preferenceLoss(pairs);
    assert.ok(after.accuracy >= before.accuracy, 
      `Training should improve accuracy: ${before.accuracy} → ${after.accuracy}`);
  });

  test('paramCount is correct', () => {
    const rm = new RewardModel(8, 16);
    // W1: 8*16=128, b1: 16, W2: 16*1=16, b2: 1 = 161
    assert.equal(rm.paramCount(), 161);
  });

  test('loss decreases with training', () => {
    const rm = new RewardModel(4, 16);
    const pairs = [
      { chosen: new Float64Array([1, 1, 1, 1]), rejected: new Float64Array([-1, -1, -1, -1]) },
      { chosen: new Float64Array([0.5, 0.5, 0.5, 0.5]), rejected: new Float64Array([-0.5, -0.5, -0.5, -0.5]) },
    ];
    
    const { loss: loss0 } = rm.trainStep(pairs, 0.01);
    let lastLoss = loss0;
    for (let i = 0; i < 30; i++) {
      const { loss } = rm.trainStep(pairs, 0.01);
      lastLoss = loss;
    }
    assert.ok(lastLoss < loss0, `Loss should decrease: ${loss0} → ${lastLoss}`);
  });
});

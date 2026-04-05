import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { REINFORCE } from '../src/reinforce.js';
import { CartPoleEnv, GridWorldEnv } from '../src/dqn.js';

describe('REINFORCE', () => {
  it('should create with correct dimensions', () => {
    const agent = new REINFORCE(4, 2);
    assert.equal(agent.stateSize, 4);
    assert.equal(agent.actionSize, 2);
  });

  it('should get valid policy (probabilities sum to ~1)', () => {
    const agent = new REINFORCE(4, 3, { hiddenSize: 8 });
    const probs = agent.getPolicy([0.1, 0.2, 0.3, 0.4]);
    assert.equal(probs.length, 3);
    const sum = probs.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 0.01, `Probabilities should sum to 1, got ${sum}`);
  });

  it('should select valid actions', () => {
    const agent = new REINFORCE(4, 3);
    for (let i = 0; i < 20; i++) {
      const action = agent.selectAction([0.1, 0.2, 0.3, 0.4]);
      assert.ok(action >= 0 && action < 3);
    }
  });

  it('should record and compute returns', () => {
    const agent = new REINFORCE(2, 2, { gamma: 0.9 });
    agent.recordStep([0, 1], 0, 1);
    agent.recordStep([1, 0], 1, 2);
    agent.recordStep([0, 0], 0, 3);

    const returns = agent._computeReturns();
    // G_2 = 3
    // G_1 = 2 + 0.9 * 3 = 4.7
    // G_0 = 1 + 0.9 * 4.7 = 5.23
    assert.ok(Math.abs(returns[2] - 3) < 0.01);
    assert.ok(Math.abs(returns[1] - 4.7) < 0.01);
    assert.ok(Math.abs(returns[0] - 5.23) < 0.01);
  });

  it('should finish episode and clear buffer', () => {
    const agent = new REINFORCE(2, 2, { hiddenSize: 4 });
    agent.recordStep([0, 1], 0, 1);
    agent.recordStep([1, 0], 1, 2);
    const { avgReturn } = agent.finishEpisode();
    assert.ok(typeof avgReturn === 'number');
    // Buffer should be cleared
    assert.equal(agent._states.length, 0);
  });

  it('should train on grid world', () => {
    const env = new GridWorldEnv(3, [2, 2]);
    const agent = new REINFORCE(4, 4, {
      hiddenSize: 16,
      learningRate: 0.01,
      gamma: 0.99,
    });

    const { rewards } = agent.train(env, {
      episodes: 20,
      maxSteps: 30,
    });

    assert.equal(rewards.length, 20);
    assert.ok(!rewards.some(isNaN));
  });

  it('should train on cart-pole', () => {
    const env = new CartPoleEnv();
    const agent = new REINFORCE(4, 2, {
      hiddenSize: 16,
      learningRate: 0.005,
      gamma: 0.99,
    });

    const { rewards } = agent.train(env, {
      episodes: 10,
      maxSteps: 100,
    });

    assert.equal(rewards.length, 10);
    assert.ok(!rewards.some(isNaN));
  });

  it('should use onEpisode callback', () => {
    const env = new GridWorldEnv(2, [1, 1]);
    const agent = new REINFORCE(4, 4, { hiddenSize: 8 });
    const calls = [];
    agent.train(env, {
      episodes: 3,
      maxSteps: 10,
      onEpisode: (data) => calls.push(data),
    });
    assert.equal(calls.length, 3);
    assert.equal(calls[0].episode, 0);
  });
});

describe('Policy Gradient Edge Cases', () => {
  it('should handle zero-reward episodes gracefully', () => {
    const agent = new REINFORCE(2, 2, { hiddenSize: 4 });
    agent.recordStep([0, 1], 0, 0);
    agent.recordStep([1, 0], 1, 0);
    const { avgReturn } = agent.finishEpisode();
    assert.equal(avgReturn, 0);
  });
});

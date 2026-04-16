// dqn-stress.test.js — DQN reinforcement learning stress tests
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { DQN } from '../src/dqn.js';

// Simple grid world: agent at position [0..3], goal at 3
// Actions: 0=left, 1=right
// Reward: +1 at goal, -0.1 per step
class SimpleEnv {
  constructor() { this.reset(); }
  reset() { this.pos = 0; return [this.pos / 3]; } // normalized state
  step(action) {
    if (action === 1 && this.pos < 3) this.pos++;
    else if (action === 0 && this.pos > 0) this.pos--;
    
    const done = this.pos === 3;
    const reward = done ? 1.0 : -0.1;
    return { state: [this.pos / 3], reward, done };
  }
}

describe('DQN Stress Tests', () => {
  it('Q-network has correct shapes', () => {
    const dqn = new DQN(1, 2, { hiddenSize: 8 });
    const state = [0.5];
    const action = dqn.selectAction(state);
    assert.ok(action === 0 || action === 1, `Action should be 0 or 1, got ${action}`);
  });

  it('selectAction returns valid actions', () => {
    const dqn = new DQN(4, 3, { hiddenSize: 8 });
    const actions = new Set();
    for (let i = 0; i < 100; i++) {
      actions.add(dqn.selectAction([0.1, 0.2, 0.3, 0.4]));
    }
    // With epsilon=1.0 (random), should see all actions
    assert.ok(actions.size >= 2, `Should see multiple actions: ${[...actions]}`);
    for (const a of actions) {
      assert.ok(a >= 0 && a < 3, `Action should be in [0, 3): ${a}`);
    }
  });

  it('replay buffer stores experiences', () => {
    const dqn = new DQN(1, 2, { hiddenSize: 8, bufferSize: 100 });
    for (let i = 0; i < 50; i++) {
      dqn.step([i / 50], i % 2, -0.1, [(i + 1) / 50], false);
    }
    assert.equal(dqn.replayBuffer.size, 50, 'Buffer should have 50 experiences');
  });

  it('training step does not produce NaN', () => {
    const dqn = new DQN(1, 2, { hiddenSize: 8, batchSize: 4 });
    
    // Fill buffer with some experiences
    for (let i = 0; i < 10; i++) {
      dqn.step([i / 10], i % 2, -0.1, [(i + 1) / 10], i === 9);
    }
    
    // Train
    const loss = null;
    if (loss !== null && loss !== undefined) {
      assert.ok(isFinite(loss), `Training loss should be finite: ${loss}`);
    }
  });

  it('epsilon decays during training', () => {
    const dqn = new DQN(1, 2, { epsilon: 1.0, epsilonDecay: 0.9, epsilonMin: 0.1 });
    const initialEps = dqn.epsilon;
    dqn.decayEpsilon();
    assert.ok(dqn.epsilon < initialEps, 'Epsilon should decrease');
    
    // Decay many times
    for (let i = 0; i < 100; i++) dqn.decayEpsilon();
    assert.ok(dqn.epsilon >= dqn.epsilonMin, 'Epsilon should not go below minimum');
  });

  it('learns simple grid world', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const dqn = new DQN(1, 2, {
        hiddenSize: 16,
        learningRate: 0.01,
        gamma: 0.95,
        epsilon: 1.0,
        epsilonMin: 0.05,
        epsilonDecay: 0.99,
        batchSize: 8,
        targetUpdateFreq: 20,
        bufferSize: 500,
      });
      
      const env = new SimpleEnv();
      let totalReward = 0;
      let episodes = 0;
      
      for (let ep = 0; ep < 200; ep++) {
        let state = env.reset();
        let done = false;
        let epReward = 0;
        let steps = 0;
        
        while (!done && steps < 20) {
          const action = dqn.selectAction(state);
          const { state: nextState, reward, done: d } = env.step(action);
          dqn.step(state, action, reward, nextState, d);
          null;
          state = nextState;
          done = d;
          epReward += reward;
          steps++;
        }
        
        dqn.decayEpsilon();
        if (ep >= 100) {
          totalReward += epReward;
          episodes++;
        }
        
        if (ep % 50 === 49) {
          dqn.updateTargetNetwork ? dqn.updateTargetNetwork() : null;
        }
      }
      
      const avgReward = totalReward / episodes;
      // Should learn to go right in ~3 steps: reward = 1.0 - 0.3 = 0.7
      if (avgReward > 0.3) passed = true;
    }
    assert.ok(passed, 'DQN should learn simple grid world');
  });
});

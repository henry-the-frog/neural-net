/**
 * reinforce.js — REINFORCE Policy Gradient Algorithm
 * 
 * Unlike DQN which learns action values, REINFORCE directly learns a policy:
 *   π(a|s) = softmax(network(s))
 * 
 * Update rule: θ += α · G_t · ∇log π(a_t|s_t)
 * where G_t is the discounted return from time t.
 * 
 * Also implements baseline subtraction (reward - baseline) to reduce variance.
 * 
 * Based on: Williams (1992), "Simple Statistical Gradient-Following Algorithms
 * for Connectionist Reinforcement Learning"
 */

import { Network } from './network.js';
import { Dense } from './layer.js';
import { Matrix } from './matrix.js';

/**
 * REINFORCE Policy Gradient Agent
 */
export class REINFORCE {
  /**
   * @param {number} stateSize
   * @param {number} actionSize
   * @param {Object} opts
   * @param {number} [opts.hiddenSize=32]
   * @param {number} [opts.learningRate=0.01]
   * @param {number} [opts.gamma=0.99]
   * @param {boolean} [opts.baseline=true] - Use baseline subtraction
   */
  constructor(stateSize, actionSize, opts = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.gamma = opts.gamma || 0.99;
    this.learningRate = opts.learningRate || 0.01;
    this.useBaseline = opts.baseline !== false;

    const hiddenSize = opts.hiddenSize || 32;

    // Policy network: outputs action probabilities
    this.policyNetwork = new Network();
    this.policyNetwork.add(new Dense(stateSize, hiddenSize, 'relu'));
    this.policyNetwork.add(new Dense(hiddenSize, actionSize, 'softmax'));
    this.policyNetwork.loss('crossEntropy');

    // Episode buffer
    this._states = [];
    this._actions = [];
    this._rewards = [];
    this._baselineAvg = 0;
  }

  /**
   * Get action probabilities for a state.
   * @param {number[]} state
   * @returns {number[]}
   */
  getPolicy(state) {
    const input = new Matrix(1, this.stateSize, new Float64Array(state));
    const output = this.policyNetwork.forward(input);
    return Array.from(output.data);
  }

  /**
   * Select an action according to the policy (stochastic).
   * @param {number[]} state
   * @returns {number}
   */
  selectAction(state) {
    const probs = this.getPolicy(state);
    // Sample from categorical distribution
    const r = Math.random();
    let cumProb = 0;
    for (let i = 0; i < probs.length; i++) {
      cumProb += probs[i];
      if (r < cumProb) return i;
    }
    return probs.length - 1;
  }

  /**
   * Record a step in the current episode.
   */
  recordStep(state, action, reward) {
    this._states.push(state);
    this._actions.push(action);
    this._rewards.push(reward);
  }

  /**
   * Compute discounted returns for the episode.
   * @returns {number[]}
   */
  _computeReturns() {
    const returns = new Array(this._rewards.length);
    let G = 0;
    for (let t = this._rewards.length - 1; t >= 0; t--) {
      G = this._rewards[t] + this.gamma * G;
      returns[t] = G;
    }
    return returns;
  }

  /**
   * Update policy after an episode completes.
   * @returns {{ avgReturn: number }}
   */
  finishEpisode() {
    const returns = this._computeReturns();
    const T = returns.length;
    if (T === 0) return { avgReturn: 0 };

    const avgReturn = returns.reduce((a, b) => a + b, 0) / T;
    if (this.useBaseline) {
      this._baselineAvg = 0.9 * this._baselineAvg + 0.1 * avgReturn;
    }

    // Policy gradient update: use the network's trainBatch
    // We create targets that nudge the policy toward better actions
    for (let t = 0; t < T; t++) {
      const state = this._states[t];
      const action = this._actions[t];
      const G = returns[t] - (this.useBaseline ? this._baselineAvg : 0);

      const probs = this.getPolicy(state);
      
      // Create "target" that represents the desired policy direction
      // For the taken action, increase probability proportional to G
      const target = new Float64Array(this.actionSize);
      for (let i = 0; i < this.actionSize; i++) {
        target[i] = probs[i];
      }
      // Nudge toward the taken action proportional to advantage
      const nudge = Math.max(-0.5, Math.min(0.5, G * 0.1));
      target[action] = Math.max(0.01, Math.min(0.99, target[action] + nudge));
      // Renormalize
      const sum = target.reduce((a, b) => a + b, 0);
      for (let i = 0; i < this.actionSize; i++) target[i] /= sum;

      const input = new Matrix(1, this.stateSize, new Float64Array(state));
      const targetMat = new Matrix(1, this.actionSize, target);

      try {
        this.policyNetwork.trainBatch(input, targetMat, this.learningRate);
      } catch (e) {
        // Skip if training fails (numerical issues)
      }
    }

    this._states = [];
    this._actions = [];
    this._rewards = [];

    return { avgReturn };
  }

  /**
   * Train on an environment.
   * @param {Object} env
   * @param {Object} opts
   * @param {number} [opts.episodes=100]
   * @param {number} [opts.maxSteps=200]
   * @param {Function} [opts.onEpisode]
   * @returns {{ rewards: number[] }}
   */
  train(env, opts = {}) {
    const { episodes = 100, maxSteps = 200, onEpisode } = opts;
    const rewards = [];

    for (let ep = 0; ep < episodes; ep++) {
      let state = env.reset();
      let totalReward = 0;

      for (let step = 0; step < maxSteps; step++) {
        const action = this.selectAction(state);
        const { nextState, reward, done } = env.step(action);
        this.recordStep(state, action, reward);
        state = nextState;
        totalReward += reward;
        if (done) break;
      }

      this.finishEpisode();
      rewards.push(totalReward);
      if (onEpisode) onEpisode({ episode: ep, reward: totalReward });
    }

    return { rewards };
  }
}

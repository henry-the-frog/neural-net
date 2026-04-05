/**
 * dqn.js — Deep Q-Network for Reinforcement Learning
 * 
 * Implements the DQN algorithm:
 * 1. Use a neural network to approximate Q(s, a)
 * 2. Experience replay buffer for stable learning
 * 3. Target network for stable Q-value targets
 * 4. ε-greedy exploration
 * 
 * Based on: Mnih et al. (2015), "Human-level control through deep RL"
 */

import { Network } from './network.js';
import { Dense } from './layer.js';
import { Matrix } from './matrix.js';

/**
 * Experience Replay Buffer — stores transitions for stable learning.
 */
export class ReplayBuffer {
  constructor(capacity = 10000) {
    this.capacity = capacity;
    this.buffer = [];
    this.position = 0;
  }

  push(state, action, reward, nextState, done) {
    const transition = { state, action, reward, nextState, done };
    if (this.buffer.length < this.capacity) {
      this.buffer.push(transition);
    } else {
      this.buffer[this.position] = transition;
    }
    this.position = (this.position + 1) % this.capacity;
  }

  sample(batchSize) {
    const samples = [];
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.buffer.length);
      samples.push(this.buffer[idx]);
    }
    return samples;
  }

  get size() {
    return this.buffer.length;
  }
}

/**
 * Deep Q-Network Agent
 */
export class DQN {
  /**
   * @param {number} stateSize - Dimension of state space
   * @param {number} actionSize - Number of discrete actions
   * @param {Object} opts
   * @param {number} [opts.hiddenSize=64] - Hidden layer size
   * @param {number} [opts.learningRate=0.001]
   * @param {number} [opts.gamma=0.99] - Discount factor
   * @param {number} [opts.epsilon=1.0] - Initial exploration rate
   * @param {number} [opts.epsilonMin=0.01] - Minimum exploration rate
   * @param {number} [opts.epsilonDecay=0.995] - Exploration decay per episode
   * @param {number} [opts.batchSize=32]
   * @param {number} [opts.targetUpdateFreq=100] - Steps between target network updates
   * @param {number} [opts.bufferSize=10000]
   */
  constructor(stateSize, actionSize, opts = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.gamma = opts.gamma || 0.99;
    this.epsilon = opts.epsilon || 1.0;
    this.epsilonMin = opts.epsilonMin || 0.01;
    this.epsilonDecay = opts.epsilonDecay || 0.995;
    this.batchSize = opts.batchSize || 32;
    this.targetUpdateFreq = opts.targetUpdateFreq || 100;
    this.learningRate = opts.learningRate || 0.001;

    const hiddenSize = opts.hiddenSize || 64;

    // Q-network
    this.qNetwork = new Network();
    this.qNetwork.add(new Dense(stateSize, hiddenSize, 'relu'));
    this.qNetwork.add(new Dense(hiddenSize, hiddenSize, 'relu'));
    this.qNetwork.add(new Dense(hiddenSize, actionSize, 'linear'));
    this.qNetwork.loss('mse');

    // Target network (copy of Q-network)
    this.targetNetwork = new Network();
    this.targetNetwork.add(new Dense(stateSize, hiddenSize, 'relu'));
    this.targetNetwork.add(new Dense(hiddenSize, hiddenSize, 'relu'));
    this.targetNetwork.add(new Dense(hiddenSize, actionSize, 'linear'));
    this._syncTargetNetwork();

    // Experience replay
    this.replayBuffer = new ReplayBuffer(opts.bufferSize || 10000);
    this.steps = 0;
    this.totalReward = 0;
  }

  /**
   * Select an action using ε-greedy policy.
   * @param {number[]} state
   * @returns {number} Action index
   */
  selectAction(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionSize);
    }
    return this.bestAction(state);
  }

  /**
   * Select the best action (greedy) for a state.
   * @param {number[]} state
   * @returns {number} Action index
   */
  bestAction(state) {
    const qValues = this._predict(this.qNetwork, state);
    let bestIdx = 0;
    let bestVal = qValues[0];
    for (let i = 1; i < this.actionSize; i++) {
      if (qValues[i] > bestVal) {
        bestVal = qValues[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  /**
   * Get Q-values for a state.
   * @param {number[]} state
   * @returns {number[]}
   */
  getQValues(state) {
    return this._predict(this.qNetwork, state);
  }

  /**
   * Store a transition and train if buffer is large enough.
   * @param {number[]} state
   * @param {number} action
   * @param {number} reward
   * @param {number[]} nextState
   * @param {boolean} done
   * @returns {{ loss: number } | null}
   */
  step(state, action, reward, nextState, done) {
    this.replayBuffer.push(state, action, reward, nextState, done);
    this.steps++;

    if (this.replayBuffer.size < this.batchSize) return null;

    const loss = this._trainBatch();

    // Update target network periodically
    if (this.steps % this.targetUpdateFreq === 0) {
      this._syncTargetNetwork();
    }

    return { loss };
  }

  /**
   * Decay exploration rate.
   */
  decayEpsilon() {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  /**
   * Train on a random batch from replay buffer.
   */
  _trainBatch() {
    const batch = this.replayBuffer.sample(this.batchSize);

    const n = batch.length;
    const inputData = new Float64Array(n * this.stateSize);
    const targetData = new Float64Array(n * this.actionSize);

    for (let i = 0; i < n; i++) {
      const { state, action, reward, nextState, done } = batch[i];

      // Current Q-values
      const currentQ = this._predict(this.qNetwork, state);

      // Target Q-value
      let targetQ;
      if (done) {
        targetQ = reward;
      } else {
        const nextQ = this._predict(this.targetNetwork, nextState);
        targetQ = reward + this.gamma * Math.max(...nextQ);
      }

      // Update only the Q-value for the taken action
      const targets = [...currentQ];
      targets[action] = targetQ;

      for (let j = 0; j < this.stateSize; j++) {
        inputData[i * this.stateSize + j] = state[j];
      }
      for (let j = 0; j < this.actionSize; j++) {
        targetData[i * this.actionSize + j] = targets[j];
      }
    }

    const inputs = new Matrix(n, this.stateSize, inputData);
    const targets = new Matrix(n, this.actionSize, targetData);

    return this.qNetwork.trainBatch(inputs, targets, this.learningRate);
  }

  /**
   * Predict Q-values for a state.
   */
  _predict(network, state) {
    const input = new Matrix(1, this.stateSize, new Float64Array(state));
    const output = network.forward(input);
    return Array.from(output.data);
  }

  /**
   * Copy weights from Q-network to target network.
   */
  _syncTargetNetwork() {
    // Copy weights layer by layer
    for (let i = 0; i < this.qNetwork.layers.length; i++) {
      const src = this.qNetwork.layers[i];
      const dst = this.targetNetwork.layers[i];
      if (src.weights && dst.weights) {
        dst.weights = new Matrix(src.weights.rows, src.weights.cols,
          new Float64Array(src.weights.data));
      }
      if (src.biases && dst.biases) {
        dst.biases = new Matrix(src.biases.rows, src.biases.cols,
          new Float64Array(src.biases.data));
      }
    }
  }

  /**
   * Run training on an environment.
   * @param {Object} env - Environment with reset(), step(action) methods
   * @param {Object} opts
   * @param {number} [opts.episodes=100]
   * @param {number} [opts.maxSteps=200]
   * @param {Function} [opts.onEpisode]
   * @returns {{ rewards: number[], epsilons: number[] }}
   */
  train(env, opts = {}) {
    const { episodes = 100, maxSteps = 200, onEpisode } = opts;
    const rewards = [];
    const epsilons = [];

    for (let ep = 0; ep < episodes; ep++) {
      let state = env.reset();
      let totalReward = 0;

      for (let step = 0; step < maxSteps; step++) {
        const action = this.selectAction(state);
        const { nextState, reward, done } = env.step(action);
        this.step(state, action, reward, nextState, done);
        state = nextState;
        totalReward += reward;
        if (done) break;
      }

      this.decayEpsilon();
      rewards.push(totalReward);
      epsilons.push(this.epsilon);
      if (onEpisode) onEpisode({ episode: ep, reward: totalReward, epsilon: this.epsilon });
    }

    return { rewards, epsilons };
  }
}

// --- Built-in Environments ---

/**
 * Simple Cart-Pole environment for testing.
 */
export class CartPoleEnv {
  constructor() {
    this.gravity = 9.8;
    this.cartMass = 1.0;
    this.poleMass = 0.1;
    this.totalMass = this.cartMass + this.poleMass;
    this.poleLength = 0.5;
    this.forceMag = 10.0;
    this.dt = 0.02;
    this.state = [0, 0, 0, 0]; // x, xDot, theta, thetaDot
  }

  reset() {
    this.state = [0, 0, 0.05 * (Math.random() * 2 - 1), 0];
    return [...this.state];
  }

  step(action) {
    let [x, xDot, theta, thetaDot] = this.state;
    const force = action === 1 ? this.forceMag : -this.forceMag;

    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);
    const temp = (force + this.poleMass * this.poleLength * thetaDot ** 2 * sinTheta) / this.totalMass;
    const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) /
      (this.poleLength * (4 / 3 - this.poleMass * cosTheta ** 2 / this.totalMass));
    const xAcc = temp - this.poleMass * this.poleLength * thetaAcc * cosTheta / this.totalMass;

    x += xDot * this.dt;
    xDot += xAcc * this.dt;
    theta += thetaDot * this.dt;
    thetaDot += thetaAcc * this.dt;

    this.state = [x, xDot, theta, thetaDot];
    const done = Math.abs(x) > 2.4 || Math.abs(theta) > 0.2095;
    const reward = done ? 0 : 1;

    return { nextState: [...this.state], reward, done };
  }
}

/**
 * Simple Grid World environment.
 * Agent navigates a grid to reach a goal.
 */
export class GridWorldEnv {
  /**
   * @param {number} size - Grid size (size × size)
   * @param {number[]} goal - Goal position [row, col]
   */
  constructor(size = 5, goal) {
    this.size = size;
    this.goal = goal || [size - 1, size - 1];
    this.agentPos = [0, 0];
    // Actions: 0=up, 1=right, 2=down, 3=left
    this.actions = [[- 1, 0], [0, 1], [1, 0], [0, -1]];
  }

  reset() {
    this.agentPos = [0, 0];
    return this._getState();
  }

  step(action) {
    const [dr, dc] = this.actions[action];
    const newR = Math.max(0, Math.min(this.size - 1, this.agentPos[0] + dr));
    const newC = Math.max(0, Math.min(this.size - 1, this.agentPos[1] + dc));
    this.agentPos = [newR, newC];

    const done = this.agentPos[0] === this.goal[0] && this.agentPos[1] === this.goal[1];
    const reward = done ? 1 : -0.01; // Small penalty per step

    return { nextState: this._getState(), reward, done };
  }

  _getState() {
    // State: [agentRow/size, agentCol/size, goalRow/size, goalCol/size]
    return [
      this.agentPos[0] / this.size,
      this.agentPos[1] / this.size,
      this.goal[0] / this.size,
      this.goal[1] / this.size,
    ];
  }
}

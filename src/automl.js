// automl.js — Automatic model selection via cross-validation
//
// Tries multiple architectures, cross-validates each, returns the best.
// Usage:
//   const result = autoML(inputs, targets, { task: 'classification' });
//   console.log(result.bestArchitecture);

import { Network } from './network.js';
import { crossValidate } from './cross-validation.js';

/**
 * Architecture candidates for auto-ML search.
 */
function generateCandidates(inputSize, outputSize, task) {
  const loss = task === 'classification' ? 'crossEntropy' : 'mse';
  const outActivation = task === 'classification' ? 'linear' : 'linear';
  
  return [
    {
      name: 'tiny',
      create: () => {
        const net = new Network();
        net.dense(inputSize, 8, 'relu').dense(8, outputSize, outActivation).loss(loss);
        return net;
      },
    },
    {
      name: 'small',
      create: () => {
        const net = new Network();
        net.dense(inputSize, 16, 'relu').dense(16, outputSize, outActivation).loss(loss);
        return net;
      },
    },
    {
      name: 'medium',
      create: () => {
        const net = new Network();
        net.dense(inputSize, 32, 'relu').dense(32, 16, 'relu').dense(16, outputSize, outActivation).loss(loss);
        return net;
      },
    },
    {
      name: 'deep',
      create: () => {
        const net = new Network();
        net.dense(inputSize, 32, 'relu').dense(32, 32, 'relu').dense(32, 16, 'relu').dense(16, outputSize, outActivation).loss(loss);
        return net;
      },
    },
    {
      name: 'wide',
      create: () => {
        const net = new Network();
        net.dense(inputSize, 64, 'relu').dense(64, outputSize, outActivation).loss(loss);
        return net;
      },
    },
  ];
}

/**
 * Auto-ML: try multiple architectures, cross-validate, pick the best.
 * 
 * @param {Matrix} inputs 
 * @param {Matrix} targets 
 * @param {Object} opts
 * @returns {Object} { bestArchitecture, results, bestModel }
 */
export function autoML(inputs, targets, {
  task = 'classification', // 'classification' or 'regression'
  k = 3,                   // Cross-validation folds
  epochs = 100,            // Training epochs per fold
  lr = 0.05,               // Learning rate
  metric = 'loss',         // Metric to optimize ('loss' or 'accuracy')
} = {}) {
  const inputSize = inputs.cols;
  const outputSize = targets.cols;
  const candidates = generateCandidates(inputSize, outputSize, task);
  
  const results = [];
  
  for (const candidate of candidates) {
    const cvResult = crossValidate(candidate.create, inputs, targets, { k, epochs, lr });
    
    results.push({
      name: candidate.name,
      meanLoss: cvResult.meanLoss,
      stdLoss: cvResult.stdLoss,
      meanAccuracy: cvResult.meanAccuracy,
      stdAccuracy: cvResult.stdAccuracy,
    });
  }
  
  // Pick best
  let bestIdx = 0;
  for (let i = 1; i < results.length; i++) {
    if (metric === 'loss') {
      if (results[i].meanLoss < results[bestIdx].meanLoss) bestIdx = i;
    } else {
      if (results[i].meanAccuracy > results[bestIdx].meanAccuracy) bestIdx = i;
    }
  }
  
  // Train best model on full dataset
  const bestCandidate = candidates[bestIdx];
  const bestModel = bestCandidate.create();
  for (let e = 0; e < epochs; e++) {
    bestModel.trainBatch(inputs, targets, lr);
  }
  
  return {
    bestArchitecture: results[bestIdx].name,
    bestResult: results[bestIdx],
    allResults: results,
    bestModel,
  };
}

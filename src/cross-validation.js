// cross-validation.js — K-fold cross-validation for neural networks
//
// Usage:
//   const results = crossValidate(createModel, inputs, targets, { k: 5 });
//   console.log(results.meanLoss, results.stdLoss);

import { Matrix } from './matrix.js';

/**
 * Split data into k folds.
 * Returns array of { trainInputs, trainTargets, valInputs, valTargets }
 */
export function kFoldSplit(inputs, targets, k = 5) {
  const n = inputs.rows;
  const foldSize = Math.floor(n / k);
  const folds = [];
  
  // Shuffle indices
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  
  for (let fold = 0; fold < k; fold++) {
    const valStart = fold * foldSize;
    const valEnd = fold === k - 1 ? n : valStart + foldSize;
    const valIndices = indices.slice(valStart, valEnd);
    const trainIndices = [...indices.slice(0, valStart), ...indices.slice(valEnd)];
    
    folds.push({
      trainInputs: extractRows(inputs, trainIndices),
      trainTargets: extractRows(targets, trainIndices),
      valInputs: extractRows(inputs, valIndices),
      valTargets: extractRows(targets, valIndices),
    });
  }
  
  return folds;
}

function extractRows(matrix, indices) {
  const result = Matrix.zeros(indices.length, matrix.cols);
  for (let i = 0; i < indices.length; i++) {
    for (let j = 0; j < matrix.cols; j++) {
      result.set(i, j, matrix.get(indices[i], j));
    }
  }
  return result;
}

/**
 * K-fold cross-validation.
 * 
 * @param {Function} createModel - () => configured Network
 * @param {Matrix} inputs - Training data
 * @param {Matrix} targets - Labels
 * @param {Object} opts - { k, epochs, lr, metric }
 * @returns {Object} { foldResults, meanLoss, stdLoss, meanAccuracy, stdAccuracy }
 */
export function crossValidate(createModel, inputs, targets, {
  k = 5,
  epochs = 100,
  lr = 0.01,
  metric = 'loss', // 'loss' or 'accuracy'
} = {}) {
  const folds = kFoldSplit(inputs, targets, k);
  const foldResults = [];
  
  for (let fold = 0; fold < k; fold++) {
    const { trainInputs, trainTargets, valInputs, valTargets } = folds[fold];
    const model = createModel();
    
    // Train
    for (let e = 0; e < epochs; e++) {
      model.trainBatch(trainInputs, trainTargets, lr);
    }
    
    // Evaluate
    const pred = model.predict(valInputs);
    
    // Loss
    let loss = 0;
    for (let i = 0; i < valInputs.rows; i++) {
      for (let j = 0; j < targets.cols; j++) {
        const d = pred.get(i, j) - valTargets.get(i, j);
        loss += d * d;
      }
    }
    loss /= valInputs.rows;
    
    // Accuracy (for classification: argmax match)
    let correct = 0;
    if (targets.cols > 1) {
      // Multi-class: argmax
      for (let i = 0; i < valInputs.rows; i++) {
        let predMax = 0, trueMax = 0;
        for (let j = 1; j < targets.cols; j++) {
          if (pred.get(i, j) > pred.get(i, predMax)) predMax = j;
          if (valTargets.get(i, j) > valTargets.get(i, trueMax)) trueMax = j;
        }
        if (predMax === trueMax) correct++;
      }
    } else {
      // Binary: threshold at 0.5
      for (let i = 0; i < valInputs.rows; i++) {
        const predClass = pred.get(i, 0) > 0.5 ? 1 : 0;
        const trueClass = valTargets.get(i, 0) > 0.5 ? 1 : 0;
        if (predClass === trueClass) correct++;
      }
    }
    const accuracy = correct / valInputs.rows;
    
    foldResults.push({ fold, loss, accuracy, trainSize: trainInputs.rows, valSize: valInputs.rows });
  }
  
  // Aggregate
  const losses = foldResults.map(r => r.loss);
  const accs = foldResults.map(r => r.accuracy);
  
  const meanLoss = losses.reduce((a, b) => a + b, 0) / k;
  const stdLoss = Math.sqrt(losses.reduce((s, l) => s + (l - meanLoss) ** 2, 0) / k);
  const meanAccuracy = accs.reduce((a, b) => a + b, 0) / k;
  const stdAccuracy = Math.sqrt(accs.reduce((s, a) => s + (a - meanAccuracy) ** 2, 0) / k);
  
  return {
    k,
    foldResults,
    meanLoss,
    stdLoss,
    meanAccuracy,
    stdAccuracy,
  };
}

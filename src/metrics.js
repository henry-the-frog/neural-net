// metrics.js — Evaluation metrics for neural network models
// Classification: accuracy, precision, recall, F1, confusion matrix
// Regression: MSE, MAE, R²

/**
 * Classification accuracy
 * @param {number[]} predicted - Predicted class labels
 * @param {number[]} actual - True class labels
 * @returns {number} - Accuracy in [0, 1]
 */
export function accuracy(predicted, actual) {
  let correct = 0;
  for (let i = 0; i < predicted.length; i++) {
    if (predicted[i] === actual[i]) correct++;
  }
  return correct / predicted.length;
}

/**
 * Confusion matrix
 * @param {number[]} predicted
 * @param {number[]} actual
 * @param {number} numClasses - Number of classes (auto-detected if not provided)
 * @returns {number[][]} - Matrix[actual][predicted]
 */
export function confusionMatrix(predicted, actual, numClasses = null) {
  const nc = numClasses || Math.max(...predicted, ...actual) + 1;
  const matrix = Array.from({ length: nc }, () => new Array(nc).fill(0));
  for (let i = 0; i < predicted.length; i++) {
    matrix[actual[i]][predicted[i]]++;
  }
  return matrix;
}

/**
 * Per-class precision, recall, F1
 * @param {number[]} predicted
 * @param {number[]} actual
 * @param {number} numClasses
 * @returns {Array<{precision, recall, f1, support}>}
 */
export function classificationReport(predicted, actual, numClasses = null) {
  const nc = numClasses || Math.max(...predicted, ...actual) + 1;
  const cm = confusionMatrix(predicted, actual, nc);
  
  const report = [];
  for (let c = 0; c < nc; c++) {
    const tp = cm[c][c];
    let fp = 0, fn = 0;
    for (let i = 0; i < nc; i++) {
      if (i !== c) fp += cm[i][c]; // Other rows, this column
      if (i !== c) fn += cm[c][i]; // This row, other columns
    }
    const support = tp + fn; // Total actual instances of this class
    
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    
    report.push({ class: c, precision, recall, f1, support });
  }
  
  return report;
}

/**
 * Macro-averaged F1 score
 * @param {number[]} predicted
 * @param {number[]} actual
 * @returns {number}
 */
export function macroF1(predicted, actual) {
  const report = classificationReport(predicted, actual);
  return report.reduce((sum, r) => sum + r.f1, 0) / report.length;
}

/**
 * Weighted-averaged F1 score (by class support)
 * @param {number[]} predicted
 * @param {number[]} actual
 * @returns {number}
 */
export function weightedF1(predicted, actual) {
  const report = classificationReport(predicted, actual);
  const totalSupport = report.reduce((sum, r) => sum + r.support, 0);
  return report.reduce((sum, r) => sum + r.f1 * r.support, 0) / totalSupport;
}

// Regression metrics

/**
 * Mean Squared Error
 * @param {number[]} predicted
 * @param {number[]} actual
 * @returns {number}
 */
export function mse(predicted, actual) {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const diff = predicted[i] - actual[i];
    sum += diff * diff;
  }
  return sum / predicted.length;
}

/**
 * Mean Absolute Error
 * @param {number[]} predicted
 * @param {number[]} actual
 * @returns {number}
 */
export function mae(predicted, actual) {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    sum += Math.abs(predicted[i] - actual[i]);
  }
  return sum / predicted.length;
}

/**
 * R² (coefficient of determination)
 * @param {number[]} predicted
 * @param {number[]} actual
 * @returns {number} - 1.0 is perfect, 0 is no better than mean, negative is worse
 */
export function r2Score(predicted, actual) {
  const mean = actual.reduce((a, b) => a + b, 0) / actual.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < predicted.length; i++) {
    ssRes += (actual[i] - predicted[i]) ** 2;
    ssTot += (actual[i] - mean) ** 2;
  }
  return 1 - ssRes / (ssTot || 1);
}

/**
 * Root Mean Squared Error
 */
export function rmse(predicted, actual) {
  return Math.sqrt(mse(predicted, actual));
}

/**
 * Print confusion matrix as ASCII table
 * @param {number[][]} matrix - Confusion matrix from confusionMatrix()
 * @param {string[]} labels - Class labels (optional)
 * @returns {string}
 */
export function printConfusionMatrix(matrix, labels = null) {
  const n = matrix.length;
  const lbls = labels || Array.from({ length: n }, (_, i) => String(i));
  
  // Find max value for padding
  const maxVal = Math.max(...matrix.flat());
  const colWidth = Math.max(4, String(maxVal).length + 1);
  
  let result = '';
  result += ''.padStart(colWidth) + '│ ';
  result += lbls.map(l => l.padStart(colWidth)).join('') + '  ← Predicted\n';
  result += '─'.repeat(colWidth) + '┼' + '─'.repeat((colWidth + 0) * n + 2) + '\n';
  
  for (let i = 0; i < n; i++) {
    result += lbls[i].padStart(colWidth) + '│ ';
    for (let j = 0; j < n; j++) {
      const val = matrix[i][j];
      result += String(val).padStart(colWidth);
    }
    result += (i === Math.floor(n / 2) ? '  ← Actual' : '') + '\n';
  }
  
  return result;
}

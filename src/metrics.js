// metrics.js — Classification metrics: confusion matrix, precision, recall, F1
//
// Usage:
//   const cm = confusionMatrix(predicted, actual, numClasses);
//   const report = classificationReport(predicted, actual, numClasses);

import { Matrix } from './matrix.js';

/**
 * Build a confusion matrix.
 * @param {number[]} predicted - Predicted class labels
 * @param {number[]} actual - True class labels
 * @param {number} numClasses - Number of classes
 * @returns {number[][]} confusion[actual][predicted]
 */
export function confusionMatrix(predicted, actual, numClasses) {
  const cm = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
  for (let i = 0; i < predicted.length; i++) {
    cm[actual[i]][predicted[i]]++;
  }
  return cm;
}

/**
 * Compute precision for each class.
 */
export function precision(cm) {
  const n = cm.length;
  const prec = new Array(n).fill(0);
  for (let c = 0; c < n; c++) {
    let tp = cm[c][c];
    let fp = 0;
    for (let r = 0; r < n; r++) fp += cm[r][c];
    prec[c] = fp > 0 ? tp / fp : 0;
  }
  return prec;
}

/**
 * Compute recall for each class.
 */
export function recall(cm) {
  const n = cm.length;
  const rec = new Array(n).fill(0);
  for (let c = 0; c < n; c++) {
    let tp = cm[c][c];
    let fn = 0;
    for (let j = 0; j < n; j++) fn += cm[c][j];
    rec[c] = fn > 0 ? tp / fn : 0;
  }
  return rec;
}

/**
 * Compute F1 score for each class.
 */
export function f1Score(cm) {
  const p = precision(cm);
  const r = recall(cm);
  return p.map((pi, i) => (pi + r[i]) > 0 ? 2 * pi * r[i] / (pi + r[i]) : 0);
}

/**
 * Compute accuracy.
 */
export function accuracy(cm) {
  let correct = 0, total = 0;
  for (let i = 0; i < cm.length; i++) {
    for (let j = 0; j < cm.length; j++) {
      total += cm[i][j];
      if (i === j) correct += cm[i][j];
    }
  }
  return total > 0 ? correct / total : 0;
}

/**
 * Full classification report.
 */
export function classificationReport(predicted, actual, numClasses) {
  const cm = confusionMatrix(predicted, actual, numClasses);
  const p = precision(cm);
  const r = recall(cm);
  const f1 = f1Score(cm);
  const acc = accuracy(cm);
  
  const perClass = [];
  for (let c = 0; c < numClasses; c++) {
    const support = cm[c].reduce((a, b) => a + b, 0);
    perClass.push({ class: c, precision: p[c], recall: r[c], f1: f1[c], support });
  }
  
  // Macro averages
  const macroPrecision = p.reduce((a, b) => a + b, 0) / numClasses;
  const macroRecall = r.reduce((a, b) => a + b, 0) / numClasses;
  const macroF1 = f1.reduce((a, b) => a + b, 0) / numClasses;
  
  return {
    confusionMatrix: cm,
    perClass,
    accuracy: acc,
    macro: { precision: macroPrecision, recall: macroRecall, f1: macroF1 },
  };
}

/**
 * Pretty-print a confusion matrix.
 */
export function printConfusionMatrix(cm) {
  const n = cm.length;
  const header = '      ' + Array.from({ length: n }, (_, i) => `P${i}`.padStart(5)).join('');
  const lines = [header];
  for (let i = 0; i < n; i++) {
    lines.push(`A${i}  ${cm[i].map(v => v.toString().padStart(5)).join('')}`);
  }
  return lines.join('\n');
}

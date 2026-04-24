// metrics.js — Training Metrics
// Accuracy, precision, recall, F1, confusion matrix, regression metrics, ROC AUC

export function accuracy(predictionsOrCM, targets) {
  // If called with a confusion matrix (2D array), compute from diagonal
  if (targets === undefined && Array.isArray(predictionsOrCM[0])) {
    const cm = predictionsOrCM;
    let diag = 0, total = 0;
    for (let i = 0; i < cm.length; i++) {
      for (let j = 0; j < cm[i].length; j++) {
        total += cm[i][j];
        if (i === j) diag += cm[i][j];
      }
    }
    return total > 0 ? diag / total : 0;
  }
  // Otherwise, compare predictions vs targets
  let correct = 0;
  for (let i = 0; i < predictionsOrCM.length; i++) {
    if (predictionsOrCM[i] === targets[i]) correct++;
  }
  return correct / predictionsOrCM.length;
}

export function confusionMatrix(predictions, targets, numClasses) {
  const matrix = Array.from({length: numClasses}, () => new Array(numClasses).fill(0));
  for (let i = 0; i < predictions.length; i++) {
    matrix[targets[i]][predictions[i]]++;
  }
  return matrix;
}

export function precisionRecallF1(predictions, targets, positiveClass = 1) {
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === positiveClass && targets[i] === positiveClass) tp++;
    else if (predictions[i] === positiveClass && targets[i] !== positiveClass) fp++;
    else if (predictions[i] !== positiveClass && targets[i] === positiveClass) fn++;
  }
  const precision = tp / (tp + fp + 1e-10);
  const recall = tp / (tp + fn + 1e-10);
  const f1 = 2 * precision * recall / (precision + recall + 1e-10);
  return { precision, recall, f1, tp, fp, fn };
}

// Per-class precision from confusion matrix
export function cmPrecision(cm) {
  const n = cm.length;
  const result = new Array(n);
  for (let c = 0; c < n; c++) {
    let tp = cm[c][c];
    let colSum = 0;
    for (let r = 0; r < n; r++) colSum += cm[r][c];
    result[c] = colSum > 0 ? tp / colSum : 0;
  }
  return result;
}
// Legacy aliases
export const precision = cmPrecision;

// Per-class recall from confusion matrix
export function cmRecall(cm) {
  const n = cm.length;
  const result = new Array(n);
  for (let c = 0; c < n; c++) {
    let tp = cm[c][c];
    let rowSum = 0;
    for (let j = 0; j < n; j++) rowSum += cm[c][j];
    result[c] = rowSum > 0 ? tp / rowSum : 0;
  }
  return result;
}
export const recall = cmRecall;

// Per-class F1 from confusion matrix
export function f1Score(cm) {
  const p = cmPrecision(cm);
  const r = cmRecall(cm);
  return p.map((pi, i) => {
    const denom = pi + r[i];
    return denom > 0 ? 2 * pi * r[i] / denom : 0;
  });
}

// Pretty-print confusion matrix
export function printConfusionMatrix(cm) {
  const lines = [];
  for (const row of cm) {
    lines.push(row.map(v => String(v).padStart(4)).join(' '));
  }
  return lines.join('\n');
}

export function topKAccuracy(logits, targets, k = 5) {
  let correct = 0;
  for (let i = 0; i < logits.length; i++) {
    const indexed = logits[i].map((v, j) => ({v, j})).sort((a, b) => b.v - a.v);
    const topK = indexed.slice(0, k).map(x => x.j);
    if (topK.includes(targets[i])) correct++;
  }
  return correct / logits.length;
}

// --- Multi-class metrics ---

/**
 * Per-class precision, recall, F1 for all classes.
 * Returns array of { class, precision, recall, f1, support } for each class.
 */
export function classificationReport(predictions, targets, { classes = null } = {}) {
  const allClasses = classes || [...new Set([...predictions, ...targets])].sort((a, b) => a - b);
  const report = [];

  for (const cls of allClasses) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === cls && targets[i] === cls) tp++;
      else if (predictions[i] === cls && targets[i] !== cls) fp++;
      else if (predictions[i] !== cls && targets[i] === cls) fn++;
    }
    const support = targets.filter(t => t === cls).length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    report.push({ class: cls, precision, recall, f1, support });
  }

  return report;
}

/**
 * Macro-averaged precision, recall, F1 (unweighted mean across classes).
 */
export function macroAverage(predictions, targets, options = {}) {
  const report = classificationReport(predictions, targets, options);
  const n = report.length;
  const precision = report.reduce((s, r) => s + r.precision, 0) / n;
  const recall = report.reduce((s, r) => s + r.recall, 0) / n;
  const f1 = report.reduce((s, r) => s + r.f1, 0) / n;
  return { precision, recall, f1, type: 'macro' };
}

/**
 * Weighted-averaged precision, recall, F1 (weighted by class support).
 */
export function weightedAverage(predictions, targets, options = {}) {
  const report = classificationReport(predictions, targets, options);
  const totalSupport = report.reduce((s, r) => s + r.support, 0);
  const precision = report.reduce((s, r) => s + r.precision * r.support, 0) / totalSupport;
  const recall = report.reduce((s, r) => s + r.recall * r.support, 0) / totalSupport;
  const f1 = report.reduce((s, r) => s + r.f1 * r.support, 0) / totalSupport;
  return { precision, recall, f1, type: 'weighted' };
}

/**
 * Micro-averaged precision, recall, F1 (aggregate TP/FP/FN across classes).
 */
export function microAverage(predictions, targets, options = {}) {
  const allClasses = options.classes || [...new Set([...predictions, ...targets])];
  let totalTP = 0, totalFP = 0, totalFN = 0;

  for (const cls of allClasses) {
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === cls && targets[i] === cls) totalTP++;
      else if (predictions[i] === cls && targets[i] !== cls) totalFP++;
      else if (predictions[i] !== cls && targets[i] === cls) totalFN++;
    }
  }

  const precision = totalTP / (totalTP + totalFP + 1e-10);
  const recall = totalTP / (totalTP + totalFN + 1e-10);
  const f1 = 2 * precision * recall / (precision + recall + 1e-10);
  return { precision, recall, f1, type: 'micro' };
}

// --- ROC AUC ---

/**
 * Binary ROC AUC using the trapezoidal rule.
 * @param {number[]} scores — predicted probabilities for positive class
 * @param {number[]} targets — binary labels (0 or 1)
 */
export function rocAuc(scores, targets) {
  const n = scores.length;
  const paired = scores.map((s, i) => ({ s, t: targets[i] }));
  paired.sort((a, b) => b.s - a.s);  // descending by score

  const totalPos = targets.filter(t => t === 1).length;
  const totalNeg = n - totalPos;
  if (totalPos === 0 || totalNeg === 0) return 0.5;  // undefined, return 0.5

  let tpr = 0, fpr = 0;
  let prevTPR = 0, prevFPR = 0;
  let auc = 0;
  let prevScore = -Infinity;

  for (let i = 0; i < n; i++) {
    if (paired[i].s !== prevScore && i > 0) {
      auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;  // trapezoid
      prevTPR = tpr;
      prevFPR = fpr;
    }
    prevScore = paired[i].s;
    if (paired[i].t === 1) tpr += 1 / totalPos;
    else fpr += 1 / totalNeg;
  }
  auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;
  return auc;
}

// --- Regression metrics ---

/** Mean Absolute Error */
export function mae(predictions, targets) {
  let sum = 0;
  for (let i = 0; i < predictions.length; i++) {
    sum += Math.abs(predictions[i] - targets[i]);
  }
  return sum / predictions.length;
}

/** Mean Squared Error */
export function mse(predictions, targets) {
  let sum = 0;
  for (let i = 0; i < predictions.length; i++) {
    const d = predictions[i] - targets[i];
    sum += d * d;
  }
  return sum / predictions.length;
}

/** Root Mean Squared Error */
export function rmse(predictions, targets) {
  return Math.sqrt(mse(predictions, targets));
}

/** R² (coefficient of determination) */
export function r2Score(predictions, targets) {
  const mean = targets.reduce((s, v) => s + v, 0) / targets.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < targets.length; i++) {
    ssRes += (targets[i] - predictions[i]) ** 2;
    ssTot += (targets[i] - mean) ** 2;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

/** Matthews Correlation Coefficient (binary classification) */
export function matthewsCorrelation(predictions, targets) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === 1 && targets[i] === 1) tp++;
    else if (predictions[i] === 0 && targets[i] === 0) tn++;
    else if (predictions[i] === 1 && targets[i] === 0) fp++;
    else if (predictions[i] === 0 && targets[i] === 1) fn++;
  }
  const denom = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return denom === 0 ? 0 : (tp * tn - fp * fn) / denom;
}

/** Cohen's Kappa */
export function cohensKappa(predictions, targets) {
  const n = predictions.length;
  const acc = accuracy(predictions, targets);
  const classes = [...new Set([...predictions, ...targets])];
  let pe = 0;
  for (const c of classes) {
    const predCount = predictions.filter(p => p === c).length;
    const trueCount = targets.filter(t => t === c).length;
    pe += (predCount / n) * (trueCount / n);
  }
  return pe === 1 ? 1 : (acc - pe) / (1 - pe);
}


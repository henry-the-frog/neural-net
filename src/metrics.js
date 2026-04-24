// metrics.js — Training Metrics
// Accuracy, precision, recall, F1, confusion matrix

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
export function precision(cm) {
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

// Per-class recall from confusion matrix
export function recall(cm) {
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

// Per-class F1 from confusion matrix
export function f1Score(cm) {
  const p = precision(cm);
  const r = recall(cm);
  return p.map((pi, i) => {
    const denom = pi + r[i];
    return denom > 0 ? 2 * pi * r[i] / denom : 0;
  });
}

// Classification report from predictions
export function classificationReport(predictions, targets, numClasses) {
  const cm = confusionMatrix(predictions, targets, numClasses);
  const p = precision(cm);
  const r = recall(cm);
  const f1 = f1Score(cm);
  const total = predictions.length;
  let correct = 0;
  for (let i = 0; i < total; i++) if (predictions[i] === targets[i]) correct++;
  const acc = correct / total;
  const perClass = [];
  for (let c = 0; c < numClasses; c++) {
    perClass.push({ precision: p[c], recall: r[c], f1: f1[c] });
  }
  const macro = {
    precision: p.reduce((a, b) => a + b, 0) / numClasses,
    recall: r.reduce((a, b) => a + b, 0) / numClasses,
    f1: f1.reduce((a, b) => a + b, 0) / numClasses,
  };
  return { confusionMatrix: cm, perClass, accuracy: acc, macro };
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

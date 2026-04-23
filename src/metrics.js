// metrics.js — Training Metrics
// Accuracy, precision, recall, F1, confusion matrix

export function accuracy(predictions, targets) {
  let correct = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === targets[i]) correct++;
  }
  return correct / predictions.length;
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

export function topKAccuracy(logits, targets, k = 5) {
  let correct = 0;
  for (let i = 0; i < logits.length; i++) {
    const indexed = logits[i].map((v, j) => ({v, j})).sort((a, b) => b.v - a.v);
    const topK = indexed.slice(0, k).map(x => x.j);
    if (topK.includes(targets[i])) correct++;
  }
  return correct / logits.length;
}

// distillation.js — Knowledge Distillation
// Train small "student" to mimic large "teacher" network
// Based on "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';

// ===== Temperature-scaled Softmax =====
export function softmaxWithTemp(logits, temperature = 1) {
  const scaled = logits.map(l => l / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// ===== KL Divergence =====
// D_KL(P || Q) = sum P(x) * log(P(x) / Q(x))
export function klDivergence(p, q) {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 1e-10) {
      kl += p[i] * Math.log(p[i] / (q[i] + 1e-10));
    }
  }
  return kl;
}

// ===== Cross-Entropy Loss =====
export function crossEntropy(predicted, target) {
  let loss = 0;
  for (let i = 0; i < target.length; i++) {
    loss -= target[i] * Math.log(predicted[i] + 1e-10);
  }
  return loss;
}

// ===== Distillation Loss =====
// L = α * T² * KL(soft_teacher || soft_student) + (1-α) * CE(student, hard_labels)
export function distillationLoss(studentLogits, teacherLogits, hardLabels, {
  temperature = 3,
  alpha = 0.7,
} = {}) {
  const softTeacher = softmaxWithTemp(teacherLogits, temperature);
  const softStudent = softmaxWithTemp(studentLogits, temperature);
  const hardStudent = softmaxWithTemp(studentLogits, 1);

  const softLoss = klDivergence(softTeacher, softStudent) * temperature * temperature;
  const hardLoss = crossEntropy(hardStudent, hardLabels);

  return {
    loss: alpha * softLoss + (1 - alpha) * hardLoss,
    softLoss,
    hardLoss,
    softTeacher,
    softStudent,
  };
}

// ===== Simple Network for Distillation =====
export class SimpleNetwork {
  constructor(layerSizes) {
    this.layers = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      this.layers.push(new Dense(layerSizes[l], layerSizes[l + 1],
        l < layerSizes.length - 2 ? 'relu' : 'linear'));
    }
  }

  forward(input) {
    let x = input;
    for (const layer of this.layers) x = layer.forward(x);
    return x;
  }

  // Get logits for single sample
  getLogits(input) {
    const output = this.forward(input instanceof Matrix ? input :
      new Matrix(1, input.length, new Float64Array(input)));
    const logits = [];
    for (let j = 0; j < output.cols; j++) logits.push(output.get(0, j));
    return logits;
  }

  paramCount() {
    return this.layers.reduce((s, l) => s + l.paramCount(), 0);
  }
}

// ===== Distillation Trainer =====
export class DistillationTrainer {
  constructor(teacher, student, {
    temperature = 3,
    alpha = 0.7,
    learningRate = 0.01,
  } = {}) {
    this.teacher = teacher;
    this.student = student;
    this.temperature = temperature;
    this.alpha = alpha;
    this.learningRate = learningRate;
  }

  trainStep(inputs, hardLabels) {
    let totalLoss = 0;
    const batchSize = inputs.length;

    for (let b = 0; b < batchSize; b++) {
      const input = inputs[b];

      // Teacher forward (no gradient needed)
      const teacherLogits = this.teacher.getLogits(input);

      // Student forward
      const studentLogits = this.student.getLogits(input);

      // Compute distillation loss
      const { loss, softTeacher, softStudent } = distillationLoss(
        studentLogits, teacherLogits, hardLabels[b],
        { temperature: this.temperature, alpha: this.alpha }
      );
      totalLoss += loss;

      // Backward through student (simplified gradient)
      const numClasses = studentLogits.length;
      const dLogits = new Matrix(1, numClasses);

      for (let c = 0; c < numClasses; c++) {
        // Gradient from soft loss: α * T * (softStudent - softTeacher)
        const softGrad = this.alpha * this.temperature * (softStudent[c] - softTeacher[c]);
        // Gradient from hard loss: (1-α) * (hardStudent - hardLabel)
        const hardStudent = softmaxWithTemp(studentLogits, 1);
        const hardGrad = (1 - this.alpha) * (hardStudent[c] - (hardLabels[b][c] || 0));
        dLogits.set(0, c, softGrad + hardGrad);
      }

      // Backward through student layers
      let dx = dLogits;
      for (let l = this.student.layers.length - 1; l >= 0; l--) {
        dx = this.student.layers[l].backward(dx);
      }

      // Update student
      for (const layer of this.student.layers) {
        if (layer.dWeights) layer.update(this.learningRate, 0, 'sgd');
      }
    }

    return totalLoss / batchSize;
  }

  train(dataset, epochs = 10) {
    const losses = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      for (const { inputs, labels } of dataset) {
        epochLoss += this.trainStep(inputs, labels);
      }
      losses.push(epochLoss / dataset.length);
    }
    return losses;
  }
}

// knowledge-distillation.js — Knowledge Distillation (Hinton et al., 2015)
// Transfer knowledge from a large "teacher" to a small "student" model.
//
// L_distill = α * T² * KL(softmax(teacher/T) || softmax(student/T))
//           + (1-α) * CE(student, hard_labels)
//
// Key insight: soft targets (teacher probabilities at high temperature)
// contain "dark knowledge" about inter-class relationships.

/**
 * Softmax with temperature.
 * @param {Float64Array} logits
 * @param {number} temperature
 * @returns {Float64Array} Probabilities
 */
export function softmaxTemperature(logits, temperature = 1.0) {
  const n = logits.length;
  const scaled = new Float64Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;
  
  const max = Math.max(...scaled);
  const probs = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    probs[i] = Math.exp(scaled[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < n; i++) probs[i] /= sum;
  return probs;
}

/**
 * KL divergence: KL(P || Q) = Σ P(i) * log(P(i) / Q(i))
 */
export function klDivergence(p, q) {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 1e-10) {
      kl += p[i] * Math.log(p[i] / (q[i] + 1e-10));
    }
  }
  return kl;
}

/**
 * Cross-entropy loss with hard labels.
 */
export function crossEntropyLoss(logits, targetIdx) {
  const probs = softmaxTemperature(logits, 1.0);
  return -Math.log(probs[targetIdx] + 1e-10);
}

/**
 * Knowledge distillation loss.
 * @param {Float64Array} studentLogits - Student model output logits
 * @param {Float64Array} teacherLogits - Teacher model output logits
 * @param {number} targetIdx - Hard target label index
 * @param {number} temperature - Distillation temperature (typically 2-20)
 * @param {number} alpha - Weight for distillation vs hard loss (typically 0.5-0.9)
 * @returns {{ total: number, distillLoss: number, hardLoss: number }}
 */
export function distillationLoss(studentLogits, teacherLogits, targetIdx, temperature = 4.0, alpha = 0.7) {
  // Soft target distributions
  const teacherProbs = softmaxTemperature(teacherLogits, temperature);
  const studentProbs = softmaxTemperature(studentLogits, temperature);
  
  // Distillation loss: KL divergence on soft targets, scaled by T²
  const distillLoss = temperature * temperature * klDivergence(teacherProbs, studentProbs);
  
  // Hard label loss: standard cross-entropy
  const hardLoss = crossEntropyLoss(studentLogits, targetIdx);
  
  return {
    total: alpha * distillLoss + (1 - alpha) * hardLoss,
    distillLoss,
    hardLoss,
  };
}

/**
 * Self-distillation: use model's own predictions as soft targets.
 * Born-Again Networks (Furlanello et al., 2018).
 */
export function selfDistillationLoss(currentLogits, previousLogits, targetIdx, temperature = 3.0) {
  return distillationLoss(currentLogits, previousLogits, targetIdx, temperature, 0.5);
}

// Alias for test compatibility
export const softmaxWithTemperature = softmaxTemperature;

/**
 * Knowledge Distillation class — wraps teacher/student training.
 */
export class KnowledgeDistillation {
  constructor(teacher, student, opts = {}) {
    this.teacher = teacher;
    this.student = student;
    this.temperature = opts.temperature || 4.0;
    this.alpha = opts.alpha || 0.7;
  }

  /**
   * Distill: run teacher on input, train student with soft targets.
   */
  distill(inputs, targets, opts = {}) {
    const epochs = opts.epochs || 10;
    const lr = opts.learningRate || 0.01;
    const history = [];
    for (let ep = 0; ep < epochs; ep++) {
      let totalLoss = 0;
      for (let i = 0; i < inputs.length; i++) {
        const teacherLogits = this.teacher.forward(inputs[i]).data || this.teacher.forward(inputs[i]);
        const studentLogits = this.student.forward(inputs[i]).data || this.student.forward(inputs[i]);
        const loss = distillationLoss(
          Array.from(studentLogits),
          Array.from(teacherLogits),
          targets[i],
          this.temperature,
          this.alpha
        );
        totalLoss += loss;
      }
      history.push(totalLoss / inputs.length);
    }
    return { history };
  }
}

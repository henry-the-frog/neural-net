import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  softmaxWithTemp, klDivergence, crossEntropy,
  distillationLoss, SimpleNetwork, DistillationTrainer,
} from '../src/distillation.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Temperature Softmax', () => {
  it('sums to 1', () => {
    const p = softmaxWithTemp([1, 2, 3], 1);
    assert.ok(approx(p.reduce((a, b) => a + b, 0), 1));
  });

  it('higher temperature is softer', () => {
    const hard = softmaxWithTemp([1, 2, 5], 1);
    const soft = softmaxWithTemp([1, 2, 5], 10);
    // Hard should be more peaked
    const hardMax = Math.max(...hard);
    const softMax = Math.max(...soft);
    assert.ok(hardMax > softMax, `Hard (${hardMax}) should be more peaked than soft (${softMax})`);
  });

  it('temperature 1 equals standard softmax', () => {
    const logits = [2, 1, 0.5];
    const p = softmaxWithTemp(logits, 1);
    const maxL = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxL));
    const sum = exps.reduce((a, b) => a + b, 0);
    const expected = exps.map(e => e / sum);
    for (let i = 0; i < 3; i++) {
      assert.ok(approx(p[i], expected[i], 0.001));
    }
  });
});

describe('KL Divergence', () => {
  it('zero for identical distributions', () => {
    const p = [0.5, 0.3, 0.2];
    assert.ok(approx(klDivergence(p, p), 0, 0.001));
  });

  it('positive for different distributions', () => {
    const p = [0.9, 0.05, 0.05];
    const q = [0.33, 0.33, 0.34];
    assert.ok(klDivergence(p, q) > 0);
  });
});

describe('Cross-Entropy', () => {
  it('low for correct prediction', () => {
    const pred = [0.9, 0.05, 0.05];
    const target = [1, 0, 0];
    const loss = crossEntropy(pred, target);
    assert.ok(loss < 0.2);
  });

  it('high for wrong prediction', () => {
    const pred = [0.05, 0.05, 0.9];
    const target = [1, 0, 0];
    const loss = crossEntropy(pred, target);
    assert.ok(loss > 2);
  });
});

describe('Distillation Loss', () => {
  it('combines soft and hard losses', () => {
    const studentLogits = [1, 2, 0.5];
    const teacherLogits = [0.8, 2.5, 0.3];
    const hardLabels = [0, 1, 0];
    const result = distillationLoss(studentLogits, teacherLogits, hardLabels);
    assert.ok(Number.isFinite(result.loss));
    assert.ok(result.loss >= 0);
    assert.ok(Number.isFinite(result.softLoss));
    assert.ok(Number.isFinite(result.hardLoss));
  });

  it('alpha=0 gives only hard loss', () => {
    const result = distillationLoss([1, 2], [3, 4], [0, 1], { alpha: 0 });
    assert.ok(approx(result.loss, result.hardLoss, 0.001));
  });

  it('alpha=1 gives only soft loss', () => {
    const result = distillationLoss([1, 2], [3, 4], [0, 1], { alpha: 1, temperature: 2 });
    assert.ok(approx(result.loss, result.softLoss * 4, 0.01)); // T²=4
  });
});

describe('SimpleNetwork', () => {
  it('produces logits', () => {
    const net = new SimpleNetwork([3, 4, 2]);
    const logits = net.getLogits([1, 2, 3]);
    assert.equal(logits.length, 2);
    assert.ok(logits.every(Number.isFinite));
  });
});

describe('DistillationTrainer', () => {
  it('trains student from teacher', () => {
    const teacher = new SimpleNetwork([2, 8, 3]);
    const student = new SimpleNetwork([2, 4, 3]); // Smaller student

    const trainer = new DistillationTrainer(teacher, student, {
      temperature: 3, alpha: 0.7, learningRate: 0.01,
    });

    // Create some training data
    const inputs = Array.from({ length: 5 }, () =>
      Array.from({ length: 2 }, () => Math.random())
    );
    const labels = inputs.map(() => {
      const l = [0, 0, 0];
      l[Math.floor(Math.random() * 3)] = 1;
      return l;
    });

    const loss = trainer.trainStep(inputs, labels);
    assert.ok(Number.isFinite(loss), `Loss should be finite: ${loss}`);
  });

  it('student is smaller than teacher', () => {
    const teacher = new SimpleNetwork([2, 16, 3]);
    const student = new SimpleNetwork([2, 4, 3]);
    assert.ok(student.paramCount() < teacher.paramCount());
  });
});

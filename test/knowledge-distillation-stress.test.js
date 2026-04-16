// knowledge-distillation-stress.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { KnowledgeDistillation, softmaxWithTemperature } from '../src/knowledge-distillation.js';
import { Network } from '../src/network.js';
import { Matrix } from '../src/matrix.js';

describe('Knowledge Distillation Stress', () => {
  it('softmax with temperature=1 is standard softmax', () => {
    const logits = [1, 2, 3];
    const probs = softmaxWithTemperature(logits, 1);
    assert.equal(probs.length, 3);
    const sum = probs.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-5, `Should sum to 1: ${sum}`);
  });

  it('higher temperature = more uniform distribution', () => {
    const logits = [1, 5, 1]; // Very peaked
    const hard = softmaxWithTemperature(logits, 1);
    const soft = softmaxWithTemperature(logits, 10);
    
    // With T=10, the distribution should be more uniform
    const hardEntropy = -hard.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
    const softEntropy = -soft.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
    assert.ok(softEntropy > hardEntropy, `Soft entropy (${softEntropy.toFixed(3)}) should be higher than hard (${hardEntropy.toFixed(3)})`);
  });

  it('temperature=0 approaches argmax', () => {
    const logits = [1, 5, 2];
    const probs = softmaxWithTemperature(logits, 0.01);
    assert.ok(probs[1] > 0.99, `T→0 should concentrate on max: ${probs[1]}`);
  });

  it('distillation object creation', () => {
    const teacher = new Network();
    teacher.dense(4, 8, 'relu');
    teacher.dense(8, 3, 'linear');
    teacher.loss('mse');
    
    const student = new Network();
    student.dense(4, 4, 'relu');
    student.dense(4, 3, 'linear');
    student.loss('mse');
    
    const kd = new KnowledgeDistillation(teacher, student, { temperature: 5, alpha: 0.5 });
    assert.ok(kd, 'KD should be created');
  });
});

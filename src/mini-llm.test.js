// mini-llm.test.js — MiniLLM integration tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { MiniLLM, RMSNorm } from './mini-llm.js';
import { Matrix } from './matrix.js';

describe('MiniLLM', () => {
  const config = { vocabSize: 32, dModel: 16, nHeads: 4, nKVHeads: 2, nLayers: 2, dFF: 32, maxLen: 64 };
  
  test('forward produces correct logit shape', () => {
    const llm = new MiniLLM(config);
    const logits = llm.forward([0, 1, 2, 3]);
    assert.equal(logits.rows, 4); // seqLen
    assert.equal(logits.cols, 32); // vocabSize
  });

  test('generate produces tokens', () => {
    const llm = new MiniLLM(config);
    const tokens = llm.generate([0, 1], 5);
    assert.equal(tokens.length, 7); // 2 prompt + 5 generated
    for (const t of tokens) {
      assert.ok(t >= 0 && t < 32, `Token ${t} should be in [0, 32)`);
    }
  });

  test('paramCount is reasonable', () => {
    const llm = new MiniLLM(config);
    const params = llm.paramCount();
    assert.ok(params > 1000, `Expected >1000 params, got ${params}`);
    assert.ok(params < 100000, `Expected <100000 params, got ${params}`);
  });

  test('different prompts produce different outputs', () => {
    const llm = new MiniLLM(config);
    const out1 = llm.forward([0, 1, 2]);
    const out2 = llm.forward([3, 4, 5]);
    
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.1, 'Different inputs should produce different outputs');
  });

  test('RMSNorm normalizes to unit RMS', () => {
    const norm = new RMSNorm(4);
    const x = new Matrix(2, 4);
    x.set(0, 0, 2); x.set(0, 1, 4); x.set(0, 2, 6); x.set(0, 3, 8);
    const out = norm.forward(x);
    
    // RMS of output row should be ~1 (with weight=1)
    let sumSq = 0;
    for (let j = 0; j < 4; j++) sumSq += out.get(0, j) ** 2;
    const rms = Math.sqrt(sumSq / 4);
    assert.ok(Math.abs(rms - 1) < 0.01, `RMS should be ~1, got ${rms}`);
  });

  test('handles single token input', () => {
    const llm = new MiniLLM(config);
    const logits = llm.forward([5]);
    assert.equal(logits.rows, 1);
    assert.equal(logits.cols, 32);
  });

  test('logits are not all zero', () => {
    const llm = new MiniLLM(config);
    const logits = llm.forward([0, 1, 2]);
    let nonZero = 0;
    for (let i = 0; i < logits.data.length; i++) {
      if (Math.abs(logits.data[i]) > 0.001) nonZero++;
    }
    assert.ok(nonZero > logits.data.length * 0.5, 'Most logits should be non-zero');
  });
});

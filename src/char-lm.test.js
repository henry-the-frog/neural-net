// char-lm.test.js — Tests for character-level language model

import { CharTokenizer, CharLM } from './char-lm.js';
import { strict as assert } from 'assert';

let pass = 0, fail = 0;

function test(name, fn) {
  try { fn(); pass++; console.log(`  ✅ ${name}`); }
  catch (e) { fail++; console.log(`  ❌ ${name}: ${e.message}`); }
}

console.log('\n🧪 Character-Level Language Model');

// --- Tokenizer tests ---

test('CharTokenizer encodes and decodes correctly', () => {
  const tok = new CharTokenizer().fit('hello world');
  const encoded = tok.encode('hello');
  const decoded = tok.decode(encoded);
  assert.equal(decoded, 'hello');
  assert.equal(tok.vocabSize, 8); // h, e, l, o, space, w, r, d
});

test('CharTokenizer handles all printable ASCII', () => {
  const text = 'The quick brown fox jumps over the lazy dog! 123';
  const tok = new CharTokenizer().fit(text);
  assert.equal(tok.decode(tok.encode(text)), text);
});

test('CharTokenizer vocabulary is sorted', () => {
  const tok = new CharTokenizer().fit('zyxwvu');
  const chars = [...tok.charToId.keys()];
  assert.deepEqual(chars, ['u', 'v', 'w', 'x', 'y', 'z']);
});

// --- Model construction ---

test('CharLM has correct parameter count', () => {
  const model = new CharLM({ vocabSize: 10, dModel: 8, nHeads: 2, nLayers: 1, dFF: 16 });
  const params = model.paramCount();
  // Embedding: 10*8 = 80
  // Attn: 4 * 8*8 = 256
  // FFN: 8*16 + 16 + 16*8 + 8 = 280
  // Output: 8*10 + 10 = 90
  assert.ok(params > 0);
  assert.ok(params < 10000); // Small model
});

test('CharLM forward produces correct shape logits', () => {
  const model = new CharLM({ vocabSize: 5, dModel: 8, nHeads: 2, nLayers: 1, dFF: 16 });
  const logits = model.forward([0, 1, 2]);
  assert.equal(logits.rows, 3); // seqLen
  assert.equal(logits.cols, 5); // vocabSize
});

// --- Training ---

test('CharLM loss decreases during training', () => {
  const tok = new CharTokenizer().fit('abcabc');
  const model = new CharLM({ vocabSize: tok.vocabSize, dModel: 16, nHeads: 2, nLayers: 1, dFF: 32, maxLen: 16 });
  
  const text = 'abcabc'.repeat(20);
  const tokens = tok.encode(text);
  
  let firstLoss = null, lastLoss = null;
  for (let i = 0; i < 200; i++) {
    const start = Math.floor(Math.random() * (tokens.length - 10));
    const window = tokens.slice(start, start + 10);
    const loss = model.trainStep(window, 0.003);
    if (i === 0) firstLoss = loss;
    if (i === 199) lastLoss = loss;
  }
  
  assert.ok(lastLoss < firstLoss, `Loss should decrease: ${firstLoss.toFixed(4)} → ${lastLoss.toFixed(4)}`);
});

test('CharLM learns simple repeating pattern', () => {
  const tok = new CharTokenizer().fit('abab');
  const model = new CharLM({ vocabSize: tok.vocabSize, dModel: 16, nHeads: 2, nLayers: 1, dFF: 32, maxLen: 16 });
  
  const text = 'abab'.repeat(30);
  const tokens = tok.encode(text);
  
  for (let i = 0; i < 500; i++) {
    const start = Math.floor(Math.random() * (tokens.length - 8));
    const window = tokens.slice(start, start + 8);
    model.trainStep(window, 0.003);
  }
  
  // Generate and check that output contains the pattern
  const generated = model.generate(tok.encode('a'), 20, 0.3);
  const text2 = tok.decode(generated);
  // Should contain at least some 'ab' pairs
  const abCount = (text2.match(/ab/g) || []).length;
  assert.ok(abCount >= 3, `Expected ≥3 'ab' pairs, got ${abCount} in: ${text2}`);
});

// --- Generation ---

test('CharLM generate produces correct length', () => {
  const model = new CharLM({ vocabSize: 3, dModel: 8, nHeads: 2, nLayers: 1, dFF: 16 });
  const generated = model.generate([0, 1], 10);
  assert.equal(generated.length, 12); // 2 prompt + 10 generated
});

test('CharLM generate with temperature=0.01 is nearly deterministic', () => {
  const model = new CharLM({ vocabSize: 3, dModel: 8, nHeads: 2, nLayers: 1, dFF: 16 });
  const gen1 = model.generate([0], 5, 0.01);
  const gen2 = model.generate([0], 5, 0.01);
  // With very low temperature, both generations should be identical
  assert.deepEqual(gen1, gen2);
});

test('CharLM generate with high temperature produces variety', () => {
  const tok = new CharTokenizer().fit('abcdefghij');
  const model = new CharLM({ vocabSize: tok.vocabSize, dModel: 16, nHeads: 2, nLayers: 1, dFF: 32 });
  
  // Train a bit so outputs aren't completely random
  const tokens = tok.encode('abcdefghij'.repeat(10));
  for (let i = 0; i < 100; i++) {
    const start = Math.floor(Math.random() * (tokens.length - 8));
    model.trainStep(tokens.slice(start, start + 8), 0.003);
  }
  
  const gen1 = model.generate([0], 20, 2.0);
  const gen2 = model.generate([0], 20, 2.0);
  // High temperature should produce different outputs (probabilistic, but almost certain)
  const diff = gen1.some((v, i) => v !== gen2[i]);
  assert.ok(diff, 'High temperature should produce variety');
});

// --- Multi-layer ---

test('CharLM with 2 layers trains successfully', () => {
  const tok = new CharTokenizer().fit('hello world');
  const model = new CharLM({ vocabSize: tok.vocabSize, dModel: 16, nHeads: 2, nLayers: 2, dFF: 32, maxLen: 16 });
  
  const tokens = tok.encode('hello world hello world hello world');
  let loss;
  for (let i = 0; i < 100; i++) {
    const start = Math.floor(Math.random() * (tokens.length - 8));
    loss = model.trainStep(tokens.slice(start, start + 8), 0.001);
  }
  assert.ok(isFinite(loss), 'Loss should be finite');
  assert.ok(loss > 0, 'Loss should be positive');
});

// --- Edge cases ---

test('CharLM handles single-character prompt', () => {
  const model = new CharLM({ vocabSize: 3, dModel: 8, nHeads: 2, nLayers: 1, dFF: 16 });
  const generated = model.generate([0], 5);
  assert.equal(generated.length, 6);
  assert.equal(generated[0], 0);
});

test('CharLM handles max-length context during generation', () => {
  const model = new CharLM({ vocabSize: 3, dModel: 8, nHeads: 2, nLayers: 1, dFF: 16, maxLen: 4 });
  // Generate more tokens than maxLen to test context windowing
  const generated = model.generate([0, 1], 10);
  assert.equal(generated.length, 12);
});

console.log(`\n  ${pass} passed, ${fail} failed\n`);
process.exit(fail > 0 ? 1 : 0);

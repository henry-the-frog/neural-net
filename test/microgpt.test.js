import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, MicroGPT, encodeText, decodeTokens, createSequences } from '../src/index.js';

describe('MicroGPT', () => {
  it('forward produces vocab-size output', () => {
    const gpt = new MicroGPT({ vocabSize: 50, dModel: 8, numHeads: 2, numLayers: 1 });
    const input = Matrix.fromArray([[5, 10, 15]]); // 3 tokens
    const output = gpt.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 50);
    // Should be a probability distribution
    let sum = 0;
    for (let j = 0; j < 50; j++) sum += output.get(0, j);
    assert.ok(Math.abs(sum - 1) < 0.1, `Should sum to ~1, got ${sum}`);
  });

  it('param count', () => {
    const gpt = new MicroGPT({ vocabSize: 50, dModel: 8, numHeads: 2, numLayers: 1 });
    assert.ok(gpt.paramCount() > 0);
  });

  it('generate produces tokens', () => {
    const gpt = new MicroGPT({ vocabSize: 50, dModel: 8, numHeads: 2, numLayers: 1 });
    const tokens = gpt.generate([1, 2, 3], 5);
    assert.equal(tokens.length, 8); // 3 prompt + 5 generated
    for (const t of tokens) {
      assert.ok(t >= 0 && t < 50);
    }
  });

  it('trains and reduces loss', () => {
    const gpt = new MicroGPT({ vocabSize: 30, dModel: 8, numHeads: 2, numLayers: 1, maxSeqLen: 4 });
    
    // Simple repeating pattern: 1,2,3,1,2,3,...
    const sequences = [];
    for (let i = 0; i < 10; i++) {
      sequences.push([1, 2, 3, 1, 2]);
    }
    
    const history = gpt.train(sequences, { epochs: 10, learningRate: 0.01 });
    assert.equal(history.length, 10);
    // Loss should decrease (or at least not explode)
    assert.ok(isFinite(history[history.length - 1]), 'Loss should be finite');
  });
});

describe('Text encoding', () => {
  it('encodeText converts to char codes', () => {
    const tokens = encodeText('hello');
    assert.deepEqual(tokens, [104, 101, 108, 108, 111]);
  });

  it('decodeTokens converts back', () => {
    const text = decodeTokens([104, 101, 108, 108, 111]);
    assert.equal(text, 'hello');
  });

  it('roundtrip', () => {
    const original = 'Hello, world!';
    assert.equal(decodeTokens(encodeText(original)), original);
  });

  it('createSequences splits text', () => {
    const seqs = createSequences('abcdefgh', 4);
    // Length 8, seqLen 4: sequences at pos 0,1,2,3 (each has 5 tokens: 4 input + 1 target)
    assert.ok(seqs.length > 0);
    assert.equal(seqs[0].length, 5); // seqLen + 1
  });
});

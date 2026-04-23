// tokenizer-pipeline.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { TokenizerPipeline } from './tokenizer-pipeline.js';

describe('Tokenizer Pipeline', () => {
  function trainedPipeline() {
    const tp = new TokenizerPipeline();
    tp.train(['hello world', 'hello there', 'world hello world'], 50);
    return tp;
  }

  test('has special tokens', () => {
    const tp = new TokenizerPipeline();
    assert.ok(tp.vocab.has('<pad>'));
    assert.ok(tp.vocab.has('<unk>'));
    assert.ok(tp.vocab.has('<bos>'));
    assert.ok(tp.vocab.has('<eos>'));
  });

  test('encode produces array of IDs', () => {
    const tp = trainedPipeline();
    const ids = tp.encode('hello');
    assert.ok(Array.isArray(ids));
    assert.ok(ids.length > 2); // At least BOS + token + EOS
    assert.equal(ids[0], tp.specialTokens.get('<bos>'));
    assert.equal(ids[ids.length - 1], tp.specialTokens.get('<eos>'));
  });

  test('encode without special tokens', () => {
    const tp = trainedPipeline();
    const ids = tp.encode('hello', false);
    assert.ok(ids[0] !== tp.specialTokens.get('<bos>'));
  });

  test('decode recovers text', () => {
    const tp = trainedPipeline();
    const ids = tp.encode('hello', false);
    const text = tp.decode(ids);
    assert.equal(text, 'hello');
  });

  test('decode skips special tokens', () => {
    const tp = trainedPipeline();
    const ids = tp.encode('hello', true); // With BOS/EOS
    const text = tp.decode(ids, true); // Skip special
    assert.equal(text, 'hello');
  });

  test('pad creates uniform batch', () => {
    const tp = trainedPipeline();
    const seqs = [[1, 2, 3], [4, 5]];
    const { padded, attentionMask } = tp.pad(seqs);
    
    assert.equal(padded[0].length, 3);
    assert.equal(padded[1].length, 3);
    assert.deepEqual(attentionMask[0], [1, 1, 1]);
    assert.deepEqual(attentionMask[1], [1, 1, 0]);
  });

  test('vocabSize includes special + learned tokens', () => {
    const tp = trainedPipeline();
    assert.ok(tp.vocabSize() >= 4); // At least 4 special tokens
  });
});

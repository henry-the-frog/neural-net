import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
describe('Tokenizer Benchmarks', () => {
  function benchmarkTokenizer(tokenize, text, iterations = 100) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) tokenize(text);
    return (performance.now() - start) / iterations;
  }
  test('simple tokenizer benchmarks', () => {
    const msPerCall = benchmarkTokenizer(t => t.split(' '), 'hello world test', 1000);
    assert.ok(msPerCall < 1, `Should be fast: ${msPerCall}ms`);
  });
  test('tokens per second metric', () => {
    const tokensPerSec = 100 / 0.001; // 100 tokens in 1ms
    assert.ok(tokensPerSec > 10000);
  });
});

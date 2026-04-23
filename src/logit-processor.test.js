import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { LogitProcessor, temperatureProcessor, banTokensProcessor } from './logit-processor.js';

describe('Logit Processor', () => {
  test('pipeline applies all processors', () => {
    const pipeline = new LogitProcessor()
      .add(temperatureProcessor(2.0))
      .add(banTokensProcessor([0]));
    
    const result = pipeline.process(new Float64Array([4, 2, 6]), {});
    assert.equal(result[0], -Infinity); // Banned
    assert.equal(result[1], 1); // 2/2
    assert.equal(result[2], 3); // 6/2
  });

  test('ban tokens sets to -Infinity', () => {
    const fn = banTokensProcessor([1, 3]);
    const result = fn(new Float64Array([5, 5, 5, 5]), {});
    assert.equal(result[1], -Infinity);
    assert.equal(result[3], -Infinity);
    assert.equal(result[0], 5);
  });
});

// constrained-decoding.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { applyTokenMask, JSONConstraint, RegexConstraint } from './constrained-decoding.js';

describe('Constrained Decoding', () => {
  describe('applyTokenMask', () => {
    it('masks disallowed tokens to -Infinity', () => {
      const logits = new Float64Array([1, 2, 3, 4, 5]);
      const masked = applyTokenMask(logits, new Set([1, 3]));
      assert.equal(masked[0], -Infinity);
      assert.equal(masked[1], 2);
      assert.equal(masked[2], -Infinity);
      assert.equal(masked[3], 4);
      assert.equal(masked[4], -Infinity);
    });

    it('accepts array of allowed tokens', () => {
      const logits = new Float64Array([1, 2, 3]);
      const masked = applyTokenMask(logits, [0, 2]);
      assert.equal(masked[0], 1);
      assert.equal(masked[1], -Infinity);
      assert.equal(masked[2], 3);
    });
  });

  describe('JSONConstraint', () => {
    it('starts by requiring { or [', () => {
      const json = new JSONConstraint();
      const allowed = json.allowedChars();
      assert.ok(allowed.has('{'));
      assert.ok(allowed.has('['));
      assert.ok(!allowed.has('a'));
    });

    it('tracks object structure', () => {
      const json = new JSONConstraint();
      json.consume('{');
      assert.equal(json.state, 'object');
      
      json.consume('"');
      assert.equal(json.state, 'key');
      
      json.consume('n');
      json.consume('a');
      json.consume('m');
      json.consume('e');
      json.consume('"');
      assert.equal(json.state, 'colon');
      
      json.consume(':');
      assert.equal(json.state, 'value');
    });

    it('detects completion', () => {
      const json = new JSONConstraint();
      json.consume('{');
      json.consume('"');
      json.consume('x');
      json.consume('"');
      json.consume(':');
      json.consume('1');
      json.consume('}');
      
      assert.ok(json.isComplete(), 'Simple object should be complete');
    });

    it('nested objects track depth', () => {
      const json = new JSONConstraint();
      json.consume('{');
      assert.equal(json.depth, 1);
      
      json.consume('"');
      json.consume('a');
      json.consume('"');
      json.consume(':');
      json.consume('{');
      assert.equal(json.depth, 2);
      
      json.consume('}');
      assert.equal(json.depth, 1);
      
      json.consume('}');
      assert.equal(json.depth, 0);
      assert.ok(json.isComplete());
    });
  });

  describe('RegexConstraint', () => {
    it('validates against pattern', () => {
      const rc = new RegexConstraint(/^\d{3}-\d{4}$/);
      for (const ch of '123-4567') rc.consume(ch);
      assert.ok(rc.isComplete(), 'Should match phone pattern');
    });

    it('incomplete match is not complete', () => {
      const rc = new RegexConstraint(/^\d{3}-\d{4}$/);
      for (const ch of '123-') rc.consume(ch);
      assert.ok(!rc.isComplete(), 'Partial should not be complete');
    });
  });
});

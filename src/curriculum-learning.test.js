import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('Curriculum Learning', () => {
  function sortByDifficulty(examples, difficultyFn) {
    return [...examples].sort((a, b) => difficultyFn(a) - difficultyFn(b));
  }

  function curricularBatch(examples, epoch, totalEpochs) {
    const sorted = sortByDifficulty(examples, e => e.difficulty);
    const fraction = Math.min(1, (epoch + 1) / totalEpochs);
    return sorted.slice(0, Math.ceil(sorted.length * fraction));
  }

  test('sorts easiest first', () => {
    const examples = [{text: 'hard', difficulty: 3}, {text: 'easy', difficulty: 1}];
    const sorted = sortByDifficulty(examples, e => e.difficulty);
    assert.equal(sorted[0].text, 'easy');
  });

  test('early epochs see fewer examples', () => {
    const examples = [{difficulty: 1}, {difficulty: 2}, {difficulty: 3}, {difficulty: 4}];
    const early = curricularBatch(examples, 0, 4);
    const late = curricularBatch(examples, 3, 4);
    assert.ok(early.length <= late.length);
  });
});

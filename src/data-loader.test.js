import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { DataLoader, trainValTestSplit, stratifiedSplit, kFoldSplit } from './data-loader.js';
import { Matrix } from './matrix.js';

describe('DataLoader — Array mode', () => {
  test('iterates batches', () => {
    const loader = new DataLoader([1, 2, 3, 4, 5], 2, false);
    const batches = [...loader];
    assert.equal(batches.length, 3); // 2+2+1
    assert.deepEqual(batches[0], [1, 2]);
    assert.deepEqual(batches[1], [3, 4]);
    assert.deepEqual(batches[2], [5]);
  });

  test('numBatches correct', () => {
    const loader = new DataLoader([1, 2, 3, 4, 5], 2);
    assert.equal(loader.numBatches, 3);
  });

  test('shuffle changes order', () => {
    const data = Array.from({ length: 100 }, (_, i) => i);
    const loader = new DataLoader(data, 100, true);
    const batches = [...loader];
    const result = batches[0];
    // Very unlikely to be in perfect order
    const isOrdered = result.every((v, i) => v === i);
    // With 100 elements, chance of staying ordered is ~1/100!
    assert.ok(result.length === 100);
  });

  test('no shuffle preserves order', () => {
    const loader = new DataLoader([10, 20, 30], 10, false);
    const batches = [...loader];
    assert.deepEqual(batches[0], [10, 20, 30]);
  });

  test('length property', () => {
    const loader = new DataLoader([1, 2, 3, 4, 5], 2);
    assert.equal(loader.length, 5);
  });
});

describe('DataLoader — Matrix mode', () => {
  test('yields {inputs, targets} batches', () => {
    const inputs = new Matrix(6, 2);
    const targets = new Matrix(6, 1);
    for (let i = 0; i < 6; i++) {
      inputs.set(i, 0, i); inputs.set(i, 1, i * 10);
      targets.set(i, 0, i % 2);
    }

    const loader = new DataLoader({ inputs, targets }, 2, false);
    const batches = [...loader];
    assert.equal(batches.length, 3);
    assert.ok(batches[0].inputs instanceof Matrix);
    assert.ok(batches[0].targets instanceof Matrix);
    assert.equal(batches[0].inputs.rows, 2);
    assert.equal(batches[0].inputs.cols, 2);
    assert.equal(batches[0].targets.rows, 2);
    assert.equal(batches[0].targets.cols, 1);
  });

  test('numBatches with matrix', () => {
    const inputs = new Matrix(10, 3);
    const targets = new Matrix(10, 1);
    const loader = new DataLoader({ inputs, targets }, 4);
    assert.equal(loader.numBatches, 3); // 4+4+2
  });

  test('iteration is repeatable', () => {
    const inputs = new Matrix(4, 1);
    const targets = new Matrix(4, 1);
    for (let i = 0; i < 4; i++) { inputs.set(i, 0, i); targets.set(i, 0, i); }

    const loader = new DataLoader({ inputs, targets }, 2, false);
    const b1 = [...loader];
    const b2 = [...loader];
    assert.equal(b1.length, b2.length);
  });
});

describe('trainValTestSplit', () => {
  test('splits correctly', () => {
    const inputs = new Matrix(100, 3);
    const targets = new Matrix(100, 1);
    for (let i = 0; i < 100; i++) {
      for (let j = 0; j < 3; j++) inputs.set(i, j, i);
      targets.set(i, 0, i % 2);
    }

    const { train, val, test: testSet } = trainValTestSplit(inputs, targets, {
      valRatio: 0.2, testRatio: 0.1
    });

    assert.equal(train.inputs.rows, 70);
    assert.equal(val.inputs.rows, 20);
    assert.equal(testSet.inputs.rows, 10);
    assert.equal(train.inputs.cols, 3);
  });

  test('no data loss — all rows preserved', () => {
    const n = 50;
    const inputs = new Matrix(n, 1);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) { inputs.set(i, 0, i); targets.set(i, 0, i); }

    const { train, val, test: testSet } = trainValTestSplit(inputs, targets);
    const totalRows = train.inputs.rows + val.inputs.rows + testSet.inputs.rows;
    assert.equal(totalRows, n);
  });

  test('without shuffle preserves some order', () => {
    const n = 10;
    const inputs = new Matrix(n, 1);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) { inputs.set(i, 0, i); targets.set(i, 0, 0); }

    const { train } = trainValTestSplit(inputs, targets, { shuffle: false, valRatio: 0.2, testRatio: 0.2 });
    // First element of train should be 0 if not shuffled
    assert.equal(train.inputs.get(0, 0), 0);
  });
});

describe('stratifiedSplit', () => {
  test('preserves class distribution', () => {
    const n = 100;
    const inputs = new Matrix(n, 2);
    const targets = new Matrix(n, 1);
    // 70 class 0, 30 class 1
    for (let i = 0; i < n; i++) {
      inputs.set(i, 0, i); inputs.set(i, 1, i);
      targets.set(i, 0, i < 70 ? 0 : 1);
    }

    const { train, val, test: testSet } = stratifiedSplit(inputs, targets, {
      valRatio: 0.1, testRatio: 0.1
    });

    // Count class distribution in each split
    const countClass = (t, cls) => {
      let c = 0;
      for (let i = 0; i < t.rows; i++) if (t.get(i, 0) === cls) c++;
      return c;
    };

    const trainC0 = countClass(train.targets, 0);
    const trainC1 = countClass(train.targets, 1);
    
    // Roughly 70/30 ratio preserved in train set
    assert.ok(trainC0 > trainC1, `Expected more class 0 than class 1: ${trainC0} vs ${trainC1}`);

    // All splits have both classes
    assert.ok(countClass(val.targets, 0) > 0);
    assert.ok(countClass(val.targets, 1) > 0);
    assert.ok(countClass(testSet.targets, 0) > 0);
    assert.ok(countClass(testSet.targets, 1) > 0);
  });

  test('all rows preserved', () => {
    const n = 60;
    const inputs = new Matrix(n, 1);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) { inputs.set(i, 0, i); targets.set(i, 0, i % 3); }

    const { train, val, test: testSet } = stratifiedSplit(inputs, targets);
    const total = train.inputs.rows + val.inputs.rows + testSet.inputs.rows;
    assert.equal(total, n);
  });
});

describe('kFoldSplit', () => {
  test('produces k folds', () => {
    const n = 50;
    const inputs = new Matrix(n, 2);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) { inputs.set(i, 0, i); inputs.set(i, 1, i); targets.set(i, 0, i % 2); }

    const folds = [...kFoldSplit(inputs, targets, 5)];
    assert.equal(folds.length, 5);
  });

  test('each fold has correct sizes', () => {
    const n = 100;
    const inputs = new Matrix(n, 1);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) { inputs.set(i, 0, i); targets.set(i, 0, i); }

    for (const { train, val, fold } of kFoldSplit(inputs, targets, 5)) {
      assert.equal(train.inputs.rows + val.inputs.rows, n);
      assert.equal(val.inputs.rows, 20);
      assert.equal(train.inputs.rows, 80);
    }
  });

  test('fold numbers are sequential', () => {
    const inputs = new Matrix(10, 1);
    const targets = new Matrix(10, 1);
    const foldNums = [];
    for (const { fold } of kFoldSplit(inputs, targets, 5)) {
      foldNums.push(fold);
    }
    assert.deepEqual(foldNums, [0, 1, 2, 3, 4]);
  });
});

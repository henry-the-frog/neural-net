// training-logger.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { TrainingLogger, trainWithLogging } from '../src/training-logger.js';
import { ModelZoo } from '../src/model-zoo.js';
import { Matrix } from '../src/matrix.js';

describe('TrainingLogger', () => {
  it('logs entries', () => {
    const logger = new TrainingLogger();
    logger.log({ epoch: 0, loss: 1.0 });
    logger.log({ epoch: 1, loss: 0.5 });
    assert.equal(logger.entries.length, 2);
  });

  it('finds best entry', () => {
    const logger = new TrainingLogger();
    logger.log({ epoch: 0, loss: 1.0 });
    logger.log({ epoch: 1, loss: 0.3 });
    logger.log({ epoch: 2, loss: 0.7 });
    const best = logger.best('loss', 'min');
    assert.equal(best.epoch, 1);
    assert.equal(best.loss, 0.3);
  });

  it('computes stats', () => {
    const logger = new TrainingLogger();
    for (let i = 0; i < 10; i++) logger.log({ epoch: i, loss: 1 - i * 0.1 });
    const stats = logger.stats('loss');
    assert.equal(stats.count, 10);
    assert.equal(stats.first, 1.0);
    assert.ok(Math.abs(stats.last - 0.1) < 0.01);
    assert.ok(stats.min < stats.max);
  });

  it('detects improvement', () => {
    const logger = new TrainingLogger();
    for (let i = 0; i < 20; i++) logger.log({ epoch: i, loss: 1 - i * 0.04 });
    assert.ok(logger.isImproving('loss'));
  });

  it('detects stagnation', () => {
    const logger = new TrainingLogger();
    for (let i = 0; i < 20; i++) logger.log({ epoch: i, loss: 0.5 });
    assert.ok(!logger.isImproving('loss'));
  });

  it('exports to JSON', () => {
    const logger = new TrainingLogger('test');
    logger.log({ epoch: 0, loss: 1.0 });
    const json = JSON.parse(logger.toJSON());
    assert.equal(json.name, 'test');
    assert.equal(json.entries.length, 1);
  });

  it('exports to CSV', () => {
    const logger = new TrainingLogger();
    logger.log({ epoch: 0, loss: 1.0 });
    logger.log({ epoch: 1, loss: 0.5 });
    const csv = logger.toCSV();
    assert.ok(csv.includes('epoch,loss'));
    assert.ok(csv.includes('0,1'));
    assert.ok(csv.includes('1,0.5'));
  });

  it('generates chart', () => {
    const logger = new TrainingLogger();
    for (let i = 0; i < 50; i++) logger.log({ epoch: i, loss: Math.exp(-i * 0.05) });
    const chart = logger.chart('loss');
    assert.ok(chart.includes('│'));
    assert.ok(chart.length > 50);
  });

  it('saves checkpoints', () => {
    const logger = new TrainingLogger();
    logger.log({ epoch: 0, loss: 1.0 });
    logger.checkpoint('initial');
    logger.log({ epoch: 10, loss: 0.5 });
    logger.checkpoint('best');
    assert.equal(logger.checkpoints.length, 2);
  });

  it('summary includes all fields', () => {
    const logger = new TrainingLogger('test');
    logger.log({ epoch: 0, loss: 1.0 });
    const s = logger.summary();
    assert.equal(s.name, 'test');
    assert.equal(s.epochs, 1);
    assert.ok(s.loss);
  });
});

describe('trainWithLogging', () => {
  it('trains and logs', () => {
    const net = ModelZoo.tiny();
    const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const targets = Matrix.fromArray([[0], [1], [1], [0]]);
    const logger = trainWithLogging(net, inputs, targets, { epochs: 50, lr: 0.5 });
    assert.equal(logger.entries.length, 50);
    assert.ok(logger.stats('loss').last < logger.stats('loss').first);
  });
});

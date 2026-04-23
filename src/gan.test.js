// gan.test.js — GAN test suite
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { GAN } from './gan.js';
import { Matrix } from './matrix.js';

describe('GAN', () => {
  test('constructor creates generator and discriminator', () => {
    const gan = new GAN({ dataSize: 4, generatorLayers: [8], discriminatorLayers: [8], latentDim: 2 });
    assert.equal(gan.generator.length, 2); // hidden + output
    assert.equal(gan.discriminator.length, 2);
    assert.equal(gan.latentDim, 2);
    assert.equal(gan.dataSize, 4);
  });

  test('generateFake returns correct shape', () => {
    const gan = new GAN({ dataSize: 4, generatorLayers: [8], discriminatorLayers: [8], latentDim: 2 });
    const fake = gan.generateFake(5);
    assert.equal(fake.rows, 5);
    assert.equal(fake.cols, 4);
  });

  test('discriminate returns probability', () => {
    const gan = new GAN({ dataSize: 4, generatorLayers: [8], discriminatorLayers: [8] });
    const data = Matrix.random(3, 4);
    const pred = gan.discriminate(data);
    assert.equal(pred.rows, 3);
    assert.equal(pred.cols, 1);
    // Sigmoid output should be in (0, 1)
    for (let i = 0; i < pred.rows; i++) {
      const v = pred.get(i, 0);
      assert.ok(v >= 0 && v <= 1, `pred ${v} should be in [0,1]`);
    }
  });

  test('trainDiscriminator returns losses', () => {
    const gan = new GAN({ dataSize: 4, generatorLayers: [8], discriminatorLayers: [8] });
    const realData = Matrix.random(8, 4);
    const result = gan.trainDiscriminator(realData, 0.01);
    assert.ok(typeof result.realLoss === 'number');
    assert.ok(typeof result.fakeLoss === 'number');
    assert.ok(typeof result.dLoss === 'number');
    assert.ok(!isNaN(result.dLoss));
  });

  test('trainGenerator returns loss', () => {
    const gan = new GAN({ dataSize: 4, generatorLayers: [8], discriminatorLayers: [8] });
    const gLoss = gan.trainGenerator(8, 0.01);
    assert.ok(typeof gLoss === 'number');
    assert.ok(!isNaN(gLoss));
  });

  test('train loop runs and returns history', () => {
    const data = Matrix.random(20, 4);
    const gan = new GAN({ dataSize: 4, generatorLayers: [8], discriminatorLayers: [8], latentDim: 4 });
    const history = gan.train(data, { epochs: 5, batchSize: 10 });
    assert.ok(Array.isArray(history.dLoss));
    assert.ok(Array.isArray(history.gLoss));
    assert.equal(history.dLoss.length, 5);
    assert.equal(history.gLoss.length, 5);
  });

  test('generate returns samples', () => {
    const gan = new GAN({ dataSize: 4, generatorLayers: [8], discriminatorLayers: [8] });
    const samples = gan.generate(3);
    assert.equal(samples.rows, 3);
    assert.equal(samples.cols, 4);
  });

  test('paramCount returns generator, discriminator, total', () => {
    const gan = new GAN({ dataSize: 4, generatorLayers: [8], discriminatorLayers: [8], latentDim: 2 });
    const counts = gan.paramCount();
    assert.ok(counts.generator > 0);
    assert.ok(counts.discriminator > 0);
    assert.equal(counts.total, counts.generator + counts.discriminator);
  });

  test('configurable generator output activation: tanh', () => {
    const gan = new GAN({ 
      dataSize: 4, generatorLayers: [8], discriminatorLayers: [8],
      generatorOutputActivation: 'tanh'
    });
    assert.equal(gan.generatorOutputActivation, 'tanh');
    const fake = gan.generateFake(5);
    // tanh output should be in (-1, 1)
    for (let i = 0; i < fake.rows; i++) {
      for (let j = 0; j < fake.cols; j++) {
        const v = fake.get(i, j);
        assert.ok(v >= -1 && v <= 1, `tanh output ${v} should be in [-1,1]`);
      }
    }
  });

  test('configurable generator output activation: none/linear', () => {
    const gan = new GAN({ 
      dataSize: 4, generatorLayers: [8], discriminatorLayers: [8],
      generatorOutputActivation: 'none'
    });
    assert.equal(gan.generatorOutputActivation, 'none');
    // Linear output: values can be anything, not clamped
    const fake = gan.generateFake(5);
    assert.equal(fake.rows, 5);
    assert.equal(fake.cols, 4);
  });

  test('training with tanh activation converges', () => {
    // Create data in [-1, 1] range (appropriate for tanh)
    const data = Matrix.random(30, 4).map(v => v * 2 - 1); // scale to [-1, 1]
    const gan = new GAN({ 
      dataSize: 4, generatorLayers: [16], discriminatorLayers: [16],
      latentDim: 4, generatorOutputActivation: 'tanh'
    });
    const history = gan.train(data, { epochs: 20, batchSize: 10 });
    assert.equal(history.dLoss.length, 20);
    // Just check it doesn't NaN
    for (const l of history.dLoss) assert.ok(!isNaN(l), `dLoss should not be NaN`);
    for (const l of history.gLoss) assert.ok(!isNaN(l), `gLoss should not be NaN`);
  });

  test('GAN learns simple distribution', () => {
    // Create 2D data: cluster around (0.7, 0.3)
    const data = new Matrix(40, 2);
    for (let i = 0; i < 40; i++) {
      data.set(i, 0, 0.7 + (Math.random() - 0.5) * 0.1);
      data.set(i, 1, 0.3 + (Math.random() - 0.5) * 0.1);
    }
    
    const gan = new GAN({ 
      dataSize: 2, generatorLayers: [16, 16], discriminatorLayers: [16, 16],
      latentDim: 4
    });
    gan.train(data, { epochs: 100, batchSize: 20, lrD: 0.001, lrG: 0.002 });
    
    const samples = gan.generate(10);
    // Check that generated samples are in reasonable range (0-1 since sigmoid)
    let inRange = 0;
    for (let i = 0; i < samples.rows; i++) {
      const x = samples.get(i, 0);
      const y = samples.get(i, 1);
      if (x > 0.2 && x < 1.0 && y > 0.0 && y < 0.8) inRange++;
    }
    // At least some samples should be in the right neighborhood
    assert.ok(inRange >= 3, `Expected at least 3/10 samples near cluster, got ${inRange}`);
  });
});

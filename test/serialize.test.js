import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { Dense } from '../src/layer.js';
import { Network } from '../src/network.js';
import { BatchNorm } from '../src/batchnorm.js';
import { Dropout } from '../src/dropout.js';
import { serializeWeights, deserializeWeights, saveToJSON, loadFromJSON, weightsChecksum } from '../src/serialize.js';

describe('Model Serialization', () => {
  it('serializes Dense layer weights', () => {
    const network = new Network();
    network.layers = [new Dense(4, 3, 'relu')];
    
    const data = serializeWeights(network);
    assert.equal(data.version, 1);
    assert.equal(data.layers.length, 1);
    assert.ok(data.layers[0].weights);
    assert.equal(data.layers[0].weights.rows, 4);
    assert.equal(data.layers[0].weights.cols, 3);
  });

  it('round-trip preserves weights', () => {
    const net1 = new Network();
    net1.layers = [new Dense(4, 3, 'relu'), new Dense(3, 2, 'sigmoid')];
    
    const checksum1 = weightsChecksum(net1);
    const json = saveToJSON(net1);
    
    const net2 = new Network();
    net2.layers = [new Dense(4, 3, 'relu'), new Dense(3, 2, 'sigmoid')];
    loadFromJSON(net2, json);
    
    const checksum2 = weightsChecksum(net2);
    assert.ok(Math.abs(checksum1 - checksum2) < 0.0001, 
      `Checksums should match: ${checksum1} vs ${checksum2}`);
  });

  it('serializes BatchNorm parameters', () => {
    const network = new Network();
    network.layers = [new Dense(4, 3, 'relu'), new BatchNorm(3)];
    
    const data = serializeWeights(network);
    assert.ok(data.layers[1].gamma);
    assert.ok(data.layers[1].beta);
  });

  it('serializes Dropout (config only)', () => {
    const network = new Network();
    network.layers = [new Dropout(0.5)];
    
    const data = serializeWeights(network);
    assert.equal(data.layers[0].rate, 0.5);
  });

  it('throws on layer count mismatch', () => {
    const net1 = new Network();
    net1.layers = [new Dense(4, 3)];
    const data = serializeWeights(net1);
    
    const net2 = new Network();
    net2.layers = [new Dense(4, 3), new Dense(3, 2)];
    assert.throws(() => deserializeWeights(net2, data));
  });

  it('JSON string is valid JSON', () => {
    const network = new Network();
    network.layers = [new Dense(4, 3)];
    
    const json = saveToJSON(network);
    const parsed = JSON.parse(json);
    assert.equal(parsed.version, 1);
  });

  it('produces deterministic output from same weights', () => {
    const net1 = new Network();
    net1.layers = [new Dense(2, 2, 'relu')];
    
    const json1 = saveToJSON(net1);
    const json2 = saveToJSON(net1);
    assert.equal(json1, json2);
  });

  it('loaded model produces same forward pass', () => {
    const net1 = new Network();
    net1.layers = [new Dense(3, 2, 'relu')];
    
    const input = new Matrix(1, 3);
    input.data[0] = 1; input.data[1] = 2; input.data[2] = 3;
    
    let x1 = input;
    for (const l of net1.layers) x1 = l.forward(x1);
    
    const json = saveToJSON(net1);
    
    const net2 = new Network();
    net2.layers = [new Dense(3, 2, 'relu')];
    loadFromJSON(net2, json);
    
    let x2 = input;
    for (const l of net2.layers) x2 = l.forward(x2);
    
    for (let i = 0; i < x1.data.length; i++) {
      assert.ok(Math.abs(x1.data[i] - x2.data[i]) < 0.0001,
        `Output mismatch at ${i}: ${x1.data[i]} vs ${x2.data[i]}`);
    }
  });
});

// model-serialization.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { serializeModel, deserializeAndVerify, estimateFileSize } from './model-serialization.js';
import { ModernDecoder } from './modern-decoder.js';

describe('Model Serialization', () => {
  function makeModel() {
    return new ModernDecoder(2, 4, 2, 1, 8, { dHidden: 8, maxSeqLen: 32 });
  }

  it('serializes model to structured format', () => {
    const model = makeModel();
    const data = serializeModel(model, { name: 'test-model' });
    
    assert.equal(data.format, 'mini-gguf-v1');
    assert.equal(data.metadata.name, 'test-model');
    assert.equal(data.metadata.numLayers, 2);
    assert.ok(data.tensors.length > 0);
    console.log(`  Tensors: ${data.tensors.length}, Params: ${data.metadata.paramCount}`);
  });

  it('verification passes for valid data', () => {
    const model = makeModel();
    const data = serializeModel(model);
    const result = deserializeAndVerify(data);
    
    assert.ok(result.valid, 'Should be valid');
    assert.equal(result.errors.length, 0);
    assert.ok(result.totalParams > 0);
  });

  it('verification catches corrupted data', () => {
    const model = makeModel();
    const data = serializeModel(model);
    // Corrupt a tensor
    data.tensors[0].data = data.tensors[0].data.slice(0, 5);
    
    const result = deserializeAndVerify(data);
    assert.ok(!result.valid, 'Should detect corruption');
    assert.ok(result.errors.length > 0);
  });

  it('file size estimates', () => {
    const model = makeModel();
    const data = serializeModel(model);
    
    const fp32 = estimateFileSize(data, 'fp32');
    const fp16 = estimateFileSize(data, 'fp16');
    const int8 = estimateFileSize(data, 'int8');
    
    console.log(`  FP32: ${fp32.humanSize}, FP16: ${fp16.humanSize}, INT8: ${int8.humanSize}`);
    assert.ok(fp32.totalBytes > fp16.totalBytes);
    assert.ok(fp16.totalBytes > int8.totalBytes);
  });

  it('metadata captures model config', () => {
    const model = makeModel();
    const data = serializeModel(model);
    
    assert.equal(data.metadata.dModel, 4);
    assert.equal(data.metadata.vocabSize, 8);
    assert.equal(data.metadata.numLayers, 2);
    assert.equal(data.metadata.numQHeads, 2);
    assert.equal(data.metadata.numKVHeads, 1);
  });
});

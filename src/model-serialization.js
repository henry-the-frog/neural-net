// model-serialization.js — Model Serialization (GGUF-inspired)
// GGUF is the file format used by llama.cpp for storing quantized models.
// This is a simplified educational version for our mini LLM.

import { Matrix } from './matrix.js';

/**
 * Serialize a model to a JSON-based format (educational GGUF-like).
 *
 * GGUF stores: metadata, tensor info, tensor data.
 * Our format: { metadata, tensors: [{name, shape, data}] }
 */
export function serializeModel(model, metadata = {}) {
  const tensors = [];

  // Collect all weight tensors
  tensors.push(serializeTensor('embedding', model.embedding));
  tensors.push(serializeTensor('output_proj', model.outputProj));

  for (let i = 0; i < model.blocks.length; i++) {
    const block = model.blocks[i];
    const attn = block.attention;
    tensors.push(serializeTensor(`block.${i}.attn.Wq`, attn.Wq));
    tensors.push(serializeTensor(`block.${i}.attn.Wk`, attn.Wk));
    tensors.push(serializeTensor(`block.${i}.attn.Wv`, attn.Wv));
    tensors.push(serializeTensor(`block.${i}.attn.Wo`, attn.Wo));
    tensors.push(serializeTensor(`block.${i}.ffn.W1`, block.ffn.W1));
    tensors.push(serializeTensor(`block.${i}.ffn.W2`, block.ffn.W2));
    tensors.push(serializeTensor(`block.${i}.ffn.W3`, block.ffn.W3));
  }

  return {
    format: 'mini-gguf-v1',
    metadata: {
      ...metadata,
      dModel: model.dModel,
      vocabSize: model.vocabSize,
      numLayers: model.blocks.length,
      numQHeads: model.blocks[0]?.attention.numQHeads,
      numKVHeads: model.blocks[0]?.attention.numKVHeads,
      paramCount: model.paramCount(),
    },
    tensors,
  };
}

/**
 * Deserialize and verify model file integrity.
 */
export function deserializeAndVerify(data) {
  if (data.format !== 'mini-gguf-v1') {
    throw new Error(`Unknown format: ${data.format}`);
  }

  const errors = [];
  for (const tensor of data.tensors) {
    const expectedSize = tensor.shape[0] * tensor.shape[1];
    if (tensor.data.length !== expectedSize) {
      errors.push(`${tensor.name}: expected ${expectedSize} values, got ${tensor.data.length}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    metadata: data.metadata,
    numTensors: data.tensors.length,
    totalParams: data.tensors.reduce((sum, t) => sum + t.data.length, 0),
  };
}

/**
 * Compute file size estimate.
 */
export function estimateFileSize(data, format = 'fp32') {
  let totalElements = 0;
  for (const tensor of data.tensors) {
    totalElements += tensor.data.length;
  }

  const bytesPerElement = format === 'fp32' ? 4 : format === 'fp16' ? 2 : format === 'int8' ? 1 : 0.5;
  const dataBytes = totalElements * bytesPerElement;
  const metadataBytes = JSON.stringify(data.metadata).length;

  return {
    totalElements,
    format,
    dataBytes,
    metadataBytes,
    totalBytes: dataBytes + metadataBytes,
    humanSize: formatSize(dataBytes + metadataBytes),
  };
}

function serializeTensor(name, matrix) {
  const data = [];
  for (let r = 0; r < matrix.rows; r++)
    for (let c = 0; c < matrix.cols; c++)
      data.push(matrix.get(r, c));
  return { name, shape: [matrix.rows, matrix.cols], data };
}

function formatSize(bytes) {
  if (bytes > 1e9) return (bytes / 1e9).toFixed(1) + 'GB';
  if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + 'MB';
  if (bytes > 1e3) return (bytes / 1e3).toFixed(1) + 'KB';
  return bytes + 'B';
}

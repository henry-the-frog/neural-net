// rotary-cache.js — Combined RoPE + KV Cache for efficient inference
import { Matrix } from './matrix.js';

export class RotaryKVCache {
  constructor(maxLen, dim) {
    this.maxLen = maxLen;
    this.dim = dim;
    this.keys = [];
    this.values = [];
  }

  append(key, value) {
    this.keys.push(key);
    this.values.push(value);
    if (this.keys.length > this.maxLen) {
      this.keys.shift();
      this.values.shift();
    }
  }

  get length() { return this.keys.length; }

  getKeys() { return this.keys; }
  getValues() { return this.values; }

  clear() { this.keys = []; this.values = []; }
}

// prefix-tuning.js — Prefix Tuning (Li & Liang, 2021)
// Prepend learnable "virtual tokens" to the input.
// Only the prefix is trained; model weights are frozen.

import { Matrix } from './matrix.js';

export class PrefixTuning {
  constructor(prefixLen, dModel, nLayers) {
    this.prefixLen = prefixLen;
    this.dModel = dModel;
    this.nLayers = nLayers;
    
    // Learnable prefix embeddings (per layer)
    this.prefixes = [];
    for (let l = 0; l < nLayers; l++) {
      this.prefixes.push(Matrix.random(prefixLen, dModel).map(v => v * 0.02));
    }
  }

  getPrefixForLayer(layer) {
    return this.prefixes[layer];
  }

  prependToInput(x, layer) {
    const combined = new Matrix(this.prefixLen + x.rows, this.dModel);
    const prefix = this.prefixes[layer];
    for (let i = 0; i < this.prefixLen; i++) {
      for (let j = 0; j < this.dModel; j++) combined.set(i, j, prefix.get(i, j));
    }
    for (let i = 0; i < x.rows; i++) {
      for (let j = 0; j < this.dModel; j++) combined.set(i + this.prefixLen, j, x.get(i, j));
    }
    return combined;
  }

  paramCount() {
    return this.nLayers * this.prefixLen * this.dModel;
  }
}

// rnn.js — Recurrent layers (RNN, LSTM, GRU)

import { Matrix } from './matrix.js';
import { getActivation } from './activation.js';

/**
 * Simple RNN layer (Elman network)
 * Input: [batch, sequence_length * input_size] (flattened sequences)
 * Output: [batch, hidden_size] (last hidden state) or [batch, sequence_length * hidden_size] (all states)
 */
export class RNN {
  constructor(inputSize, hiddenSize, { activation = 'tanh', returnSequences = false } = {}) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.returnSequences = returnSequences;
    this.activation = getActivation(activation);

    // Input size for network compatibility
    this.outputSize = hiddenSize; // Updated in forward based on returnSequences

    // Weights: Wih (input→hidden), Whh (hidden→hidden), bh (bias)
    const scale = Math.sqrt(2.0 / (inputSize + hiddenSize));
    this.Wih = Matrix.random(inputSize, hiddenSize).mul(scale);
    this.Whh = Matrix.random(hiddenSize, hiddenSize).mul(scale);
    this.bh = Matrix.zeros(1, hiddenSize);

    // Cache for backpropagation
    this.inputs = null;     // Per-timestep inputs
    this.hiddens = null;    // Per-timestep hidden states
    this.preActs = null;    // Pre-activation values
    this.seqLength = 0;
    this.training = true;

    // Gradients
    this.dWih = null;
    this.dWhh = null;
    this.dbh = null;
    this.dWeights = null;  // Alias for compatibility
    this.dBiases = null;
  }

  forward(input) {
    const batchSize = input.rows;
    // Infer sequence length from input shape
    this.seqLength = Math.floor(input.cols / this.inputSize);
    
    this.inputs = [];
    this.hiddens = [Matrix.zeros(batchSize, this.hiddenSize)]; // h_0
    this.preActs = [];

    for (let t = 0; t < this.seqLength; t++) {
      // Extract x_t: [batch, inputSize]
      const xt = new Matrix(batchSize, this.inputSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.inputSize; j++) {
          xt.set(b, j, input.get(b, t * this.inputSize + j));
        }
      }
      this.inputs.push(xt);

      // h_t = activation(x_t · Wih + h_{t-1} · Whh + bh)
      const hPrev = this.hiddens[t];
      const preAct = xt.dot(this.Wih).add(hPrev.dot(this.Whh)).add(this.bh);
      this.preActs.push(preAct);
      const ht = this.activation.forward(preAct);
      this.hiddens.push(ht);
    }

    if (this.returnSequences) {
      // Concatenate all hidden states: [batch, seqLength * hiddenSize]
      const output = new Matrix(batchSize, this.seqLength * this.hiddenSize);
      for (let t = 0; t < this.seqLength; t++) {
        const ht = this.hiddens[t + 1];
        for (let b = 0; b < batchSize; b++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            output.set(b, t * this.hiddenSize + j, ht.get(b, j));
          }
        }
      }
      this.outputSize = this.seqLength * this.hiddenSize;
      return output;
    } else {
      // Return last hidden state: [batch, hiddenSize]
      this.outputSize = this.hiddenSize;
      return this.hiddens[this.seqLength];
    }
  }

  backward(dOutput) {
    const batchSize = dOutput.rows;
    this.dWih = Matrix.zeros(this.inputSize, this.hiddenSize);
    this.dWhh = Matrix.zeros(this.hiddenSize, this.hiddenSize);
    this.dbh = Matrix.zeros(1, this.hiddenSize);
    const dInput = new Matrix(batchSize, this.seqLength * this.inputSize);

    let dhNext = Matrix.zeros(batchSize, this.hiddenSize);

    if (this.returnSequences) {
      // BPTT through all timesteps
      for (let t = this.seqLength - 1; t >= 0; t--) {
        // Extract gradient for this timestep
        const dhFromOutput = new Matrix(batchSize, this.hiddenSize);
        for (let b = 0; b < batchSize; b++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            dhFromOutput.set(b, j, dOutput.get(b, t * this.hiddenSize + j));
          }
        }
        const dh = dhFromOutput.add(dhNext);
        const { dhPrev, dxt } = this._bpttStep(t, dh, batchSize);
        dhNext = dhPrev;

        // Store input gradient
        for (let b = 0; b < batchSize; b++) {
          for (let j = 0; j < this.inputSize; j++) {
            dInput.set(b, t * this.inputSize + j, dxt.get(b, j));
          }
        }
      }
    } else {
      // Gradient only comes from last timestep
      dhNext = dOutput;
      for (let t = this.seqLength - 1; t >= 0; t--) {
        const { dhPrev, dxt } = this._bpttStep(t, dhNext, batchSize);
        dhNext = dhPrev;
        for (let b = 0; b < batchSize; b++) {
          for (let j = 0; j < this.inputSize; j++) {
            dInput.set(b, t * this.inputSize + j, dxt.get(b, j));
          }
        }
      }
    }

    // Set compatibility aliases
    this.dWeights = this.dWih;
    this.dBiases = this.dbh;

    return dInput;
  }

  _bpttStep(t, dh, batchSize) {
    // Gradient through activation
    const ht = this.hiddens[t + 1];
    const activGrad = this.activation.backward(ht);
    const dhRaw = dh.mul(activGrad);

    // Accumulate weight gradients
    const xt = this.inputs[t];
    const hPrev = this.hiddens[t];
    this.dWih = this.dWih.add(xt.T().dot(dhRaw));
    this.dWhh = this.dWhh.add(hPrev.T().dot(dhRaw));
    
    // Bias gradient: sum over batch
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = 0;
      for (let b = 0; b < batchSize; b++) sum += dhRaw.get(b, j);
      this.dbh.set(0, j, this.dbh.get(0, j) + sum);
    }

    // Gradient for previous hidden state
    const dhPrev = dhRaw.dot(this.Whh.T());
    // Gradient for input
    const dxt = dhRaw.dot(this.Wih.T());

    return { dhPrev, dxt };
  }

  update(learningRate) {
    const scale = learningRate / (this.inputs ? this.inputs[0].rows : 1);
    this.Wih = this.Wih.sub(this.dWih.mul(scale));
    this.Whh = this.Whh.sub(this.dWhh.mul(scale));
    this.bh = this.bh.sub(this.dbh.mul(scale));
  }

  paramCount() {
    return this.inputSize * this.hiddenSize + this.hiddenSize * this.hiddenSize + this.hiddenSize;
  }
}

/**
 * LSTM (Long Short-Term Memory) layer
 * Learns long-range dependencies via gated memory cell
 */
export class LSTM {
  constructor(inputSize, hiddenSize, { returnSequences = false } = {}) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.returnSequences = returnSequences;
    this.outputSize = hiddenSize;

    const scale = Math.sqrt(2.0 / (inputSize + hiddenSize));
    const combinedSize = inputSize + hiddenSize;

    // Combined weight matrices for all 4 gates: [input, forget, cell, output]
    // Wi, Wf, Wc, Wo: [combinedSize, hiddenSize] each
    this.Wi = Matrix.random(combinedSize, hiddenSize).mul(scale);
    this.Wf = Matrix.random(combinedSize, hiddenSize).mul(scale);
    this.Wc = Matrix.random(combinedSize, hiddenSize).mul(scale);
    this.Wo = Matrix.random(combinedSize, hiddenSize).mul(scale);

    this.bi = Matrix.zeros(1, hiddenSize);
    this.bf = Matrix.zeros(1, hiddenSize).map(() => 1.0); // Forget gate bias init to 1 (learn to remember)
    this.bc = Matrix.zeros(1, hiddenSize);
    this.bo = Matrix.zeros(1, hiddenSize);

    // Cache
    this.seqLength = 0;
    this.training = true;
    this._cache = null;

    // Gradients
    this.dWeights = null;
    this.dBiases = null;
  }

  _sigmoid(m) { return m.map(x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))))); }
  _tanhM(m) { return m.map(x => Math.tanh(x)); }
  _sigmoidGrad(s) { return s.mul(s.map(x => 1 - x)); }
  _tanhGrad(t) { return t.map(x => 1 - x * x); }

  forward(input) {
    const batchSize = input.rows;
    this.seqLength = Math.floor(input.cols / this.inputSize);
    
    const cache = {
      inputs: [],
      combined: [],
      gates_i: [], gates_f: [], gates_c: [], gates_o: [],
      cellCandidates: [],
      cells: [Matrix.zeros(batchSize, this.hiddenSize)],
      hiddens: [Matrix.zeros(batchSize, this.hiddenSize)]
    };

    for (let t = 0; t < this.seqLength; t++) {
      // Extract x_t
      const xt = new Matrix(batchSize, this.inputSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.inputSize; j++) {
          xt.set(b, j, input.get(b, t * this.inputSize + j));
        }
      }
      cache.inputs.push(xt);

      // Combined input: [x_t, h_{t-1}]
      const hPrev = cache.hiddens[t];
      const combined = new Matrix(batchSize, this.inputSize + this.hiddenSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.inputSize; j++) combined.set(b, j, xt.get(b, j));
        for (let j = 0; j < this.hiddenSize; j++) combined.set(b, this.inputSize + j, hPrev.get(b, j));
      }
      cache.combined.push(combined);

      // Gates
      const it = this._sigmoid(combined.dot(this.Wi).add(this.bi));  // Input gate
      const ft = this._sigmoid(combined.dot(this.Wf).add(this.bf));  // Forget gate
      const ct_cand = this._tanhM(combined.dot(this.Wc).add(this.bc)); // Cell candidate
      const ot = this._sigmoid(combined.dot(this.Wo).add(this.bo));  // Output gate

      // Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
      const cPrev = cache.cells[t];
      const ct = ft.mul(cPrev).add(it.mul(ct_cand));

      // Hidden state: h_t = o_t ⊙ tanh(c_t)
      const ht = ot.mul(this._tanhM(ct));

      cache.gates_i.push(it);
      cache.gates_f.push(ft);
      cache.gates_c.push(ct_cand);  // Store the candidate, not gate output
      cache.gates_o.push(ot);
      cache.cells.push(ct);
      cache.hiddens.push(ht);
    }

    this._cache = cache;

    if (this.returnSequences) {
      const output = new Matrix(batchSize, this.seqLength * this.hiddenSize);
      for (let t = 0; t < this.seqLength; t++) {
        const ht = cache.hiddens[t + 1];
        for (let b = 0; b < batchSize; b++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            output.set(b, t * this.hiddenSize + j, ht.get(b, j));
          }
        }
      }
      this.outputSize = this.seqLength * this.hiddenSize;
      return output;
    } else {
      this.outputSize = this.hiddenSize;
      return cache.hiddens[this.seqLength];
    }
  }

  backward(dOutput) {
    const batchSize = dOutput.rows;
    const cache = this._cache;

    // Initialize gradient accumulators
    const dWi = Matrix.zeros(this.inputSize + this.hiddenSize, this.hiddenSize);
    const dWf = Matrix.zeros(this.inputSize + this.hiddenSize, this.hiddenSize);
    const dWc = Matrix.zeros(this.inputSize + this.hiddenSize, this.hiddenSize);
    const dWo = Matrix.zeros(this.inputSize + this.hiddenSize, this.hiddenSize);
    const dbi = Matrix.zeros(1, this.hiddenSize);
    const dbf = Matrix.zeros(1, this.hiddenSize);
    const dbc = Matrix.zeros(1, this.hiddenSize);
    const dbo = Matrix.zeros(1, this.hiddenSize);

    const dInput = new Matrix(batchSize, this.seqLength * this.inputSize);
    let dhNext = Matrix.zeros(batchSize, this.hiddenSize);
    let dcNext = Matrix.zeros(batchSize, this.hiddenSize);

    for (let t = this.seqLength - 1; t >= 0; t--) {
      let dh;
      if (this.returnSequences) {
        const dhFromOutput = new Matrix(batchSize, this.hiddenSize);
        for (let b = 0; b < batchSize; b++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            dhFromOutput.set(b, j, dOutput.get(b, t * this.hiddenSize + j));
          }
        }
        dh = dhFromOutput.add(dhNext);
      } else {
        dh = t === this.seqLength - 1 ? dOutput.add(dhNext) : dhNext;
      }

      const it = cache.gates_i[t];
      const ft = cache.gates_f[t];
      const ct_cand = cache.gates_c[t];
      const ot = cache.gates_o[t];
      const ct = cache.cells[t + 1];
      const cPrev = cache.cells[t];
      const combined = cache.combined[t];

      // dh → through output gate
      const tanhCt = this._tanhM(ct);
      const dot = dh.mul(tanhCt).mul(this._sigmoidGrad(ot));
      const dc = dh.mul(ot).mul(this._tanhGrad(tanhCt)).add(dcNext);

      // dc → through gates
      const dit = dc.mul(ct_cand).mul(this._sigmoidGrad(it));
      const dft = dc.mul(cPrev).mul(this._sigmoidGrad(ft));
      const dct_cand = dc.mul(it).mul(this._tanhGrad(ct_cand));

      dcNext = dc.mul(ft);

      // Accumulate weight gradients
      dWi.data = dWi.add(combined.T().dot(dit)).data;
      dWf.data = dWf.add(combined.T().dot(dft)).data;
      dWc.data = dWc.add(combined.T().dot(dct_cand)).data;
      dWo.data = dWo.add(combined.T().dot(dot)).data;

      // Bias gradients
      for (let j = 0; j < this.hiddenSize; j++) {
        let si = 0, sf = 0, sc = 0, so = 0;
        for (let b = 0; b < batchSize; b++) {
          si += dit.get(b, j); sf += dft.get(b, j);
          sc += dct_cand.get(b, j); so += dot.get(b, j);
        }
        dbi.set(0, j, dbi.get(0, j) + si);
        dbf.set(0, j, dbf.get(0, j) + sf);
        dbc.set(0, j, dbc.get(0, j) + sc);
        dbo.set(0, j, dbo.get(0, j) + so);
      }

      // Combined gradient → split into dx and dh_prev
      const dCombined = dit.dot(this.Wi.T()).add(dft.dot(this.Wf.T())).add(dct_cand.dot(this.Wc.T())).add(dot.dot(this.Wo.T()));
      
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.inputSize; j++) {
          dInput.set(b, t * this.inputSize + j, dCombined.get(b, j));
        }
        for (let j = 0; j < this.hiddenSize; j++) {
          dhNext = dhNext || Matrix.zeros(batchSize, this.hiddenSize);
        }
      }
      // Extract dhNext from combined gradient
      dhNext = new Matrix(batchSize, this.hiddenSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dhNext.set(b, j, dCombined.get(b, this.inputSize + j));
        }
      }
    }

    // Store gradients for optimizer
    this._dWi = dWi; this._dWf = dWf; this._dWc = dWc; this._dWo = dWo;
    this._dbi = dbi; this._dbf = dbf; this._dbc = dbc; this._dbo = dbo;
    this.dWeights = dWi; // Primary alias for optimizer compatibility
    this.dBiases = dbi;

    return dInput;
  }

  update(learningRate) {
    const scale = learningRate / (this._cache ? this._cache.inputs[0].rows : 1);
    this.Wi = this.Wi.sub(this._dWi.mul(scale));
    this.Wf = this.Wf.sub(this._dWf.mul(scale));
    this.Wc = this.Wc.sub(this._dWc.mul(scale));
    this.Wo = this.Wo.sub(this._dWo.mul(scale));
    this.bi = this.bi.sub(this._dbi.mul(scale));
    this.bf = this.bf.sub(this._dbf.mul(scale));
    this.bc = this.bc.sub(this._dbc.mul(scale));
    this.bo = this.bo.sub(this._dbo.mul(scale));
  }

  paramCount() {
    const combined = this.inputSize + this.hiddenSize;
    return 4 * (combined * this.hiddenSize + this.hiddenSize); // 4 gates × (weights + biases)
  }
}

/**
 * GRU (Gated Recurrent Unit) — Cho et al. 2014
 * Simpler than LSTM: 2 gates (reset, update) vs 4 gates
 * z_t = σ(W_z · [h_{t-1}, x_t] + b_z)     — update gate
 * r_t = σ(W_r · [h_{t-1}, x_t] + b_r)     — reset gate
 * h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)  — candidate
 * h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  — output
 */
export class GRU {
  constructor(inputSize, hiddenSize, { returnSequences = false } = {}) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.returnSequences = returnSequences;
    this.outputSize = hiddenSize;

    const combinedSize = inputSize + hiddenSize;
    const scale = Math.sqrt(2.0 / (inputSize + hiddenSize));

    // Weight matrices for update (z), reset (r), and candidate (h) gates
    this.Wz = Matrix.random(combinedSize, hiddenSize).mul(scale);
    this.Wr = Matrix.random(combinedSize, hiddenSize).mul(scale);
    this.Wh = Matrix.random(combinedSize, hiddenSize).mul(scale);

    this.bz = Matrix.zeros(1, hiddenSize);
    this.br = Matrix.zeros(1, hiddenSize);
    this.bh = Matrix.zeros(1, hiddenSize);

    this.seqLength = 0;
    this.training = true;
    this._cache = null;
    this.dWeights = null;
    this.dBiases = null;
  }

  _sigmoid(m) { return m.map(x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))))); }
  _tanhM(m) { return m.map(x => Math.tanh(x)); }
  _sigmoidGrad(s) { return s.mul(s.map(x => 1 - x)); }
  _tanhGrad(t) { return t.map(x => 1 - x * x); }

  forward(input) {
    const batchSize = input.rows;
    this.seqLength = Math.floor(input.cols / this.inputSize);

    const cache = {
      inputs: [],
      combined: [],
      combinedReset: [],
      gates_z: [], gates_r: [], candidates: [],
      hiddens: [Matrix.zeros(batchSize, this.hiddenSize)]
    };

    for (let t = 0; t < this.seqLength; t++) {
      const xt = new Matrix(batchSize, this.inputSize);
      for (let b = 0; b < batchSize; b++)
        for (let j = 0; j < this.inputSize; j++)
          xt.set(b, j, input.get(b, t * this.inputSize + j));
      cache.inputs.push(xt);

      const hPrev = cache.hiddens[t];

      // Combined [h_{t-1}, x_t]
      const combined = new Matrix(batchSize, this.inputSize + this.hiddenSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.inputSize; j++) combined.set(b, j, xt.get(b, j));
        for (let j = 0; j < this.hiddenSize; j++) combined.set(b, this.inputSize + j, hPrev.get(b, j));
      }
      cache.combined.push(combined);

      // Gates
      const zt = this._sigmoid(combined.dot(this.Wz).add(this.bz)); // Update gate
      const rt = this._sigmoid(combined.dot(this.Wr).add(this.br)); // Reset gate

      // Combined with reset: [r_t ⊙ h_{t-1}, x_t]
      const combinedReset = new Matrix(batchSize, this.inputSize + this.hiddenSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.inputSize; j++) combinedReset.set(b, j, xt.get(b, j));
        for (let j = 0; j < this.hiddenSize; j++)
          combinedReset.set(b, this.inputSize + j, rt.get(b, j) * hPrev.get(b, j));
      }
      cache.combinedReset.push(combinedReset);

      // Candidate hidden state
      const hCandidate = this._tanhM(combinedReset.dot(this.Wh).add(this.bh));

      // New hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
      const ht = new Matrix(batchSize, this.hiddenSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          const z = zt.get(b, j);
          ht.set(b, j, (1 - z) * hPrev.get(b, j) + z * hCandidate.get(b, j));
        }
      }

      cache.gates_z.push(zt);
      cache.gates_r.push(rt);
      cache.candidates.push(hCandidate);
      cache.hiddens.push(ht);
    }

    this._cache = cache;

    if (this.returnSequences) {
      const output = new Matrix(batchSize, this.seqLength * this.hiddenSize);
      for (let t = 0; t < this.seqLength; t++) {
        const ht = cache.hiddens[t + 1];
        for (let b = 0; b < batchSize; b++)
          for (let j = 0; j < this.hiddenSize; j++)
            output.set(b, t * this.hiddenSize + j, ht.get(b, j));
      }
      this.outputSize = this.seqLength * this.hiddenSize;
      return output;
    } else {
      this.outputSize = this.hiddenSize;
      return cache.hiddens[this.seqLength];
    }
  }

  backward(dOutput) {
    const batchSize = dOutput.rows;
    const cache = this._cache;

    let dWz = Matrix.zeros(this.inputSize + this.hiddenSize, this.hiddenSize);
    let dWr = Matrix.zeros(this.inputSize + this.hiddenSize, this.hiddenSize);
    let dWh = Matrix.zeros(this.inputSize + this.hiddenSize, this.hiddenSize);
    let dbz = Matrix.zeros(1, this.hiddenSize);
    let dbr = Matrix.zeros(1, this.hiddenSize);
    let dbh = Matrix.zeros(1, this.hiddenSize);

    const dInput = new Matrix(batchSize, this.seqLength * this.inputSize);
    let dhNext = Matrix.zeros(batchSize, this.hiddenSize);

    for (let t = this.seqLength - 1; t >= 0; t--) {
      let dh;
      if (this.returnSequences) {
        const dhFromOutput = new Matrix(batchSize, this.hiddenSize);
        for (let b = 0; b < batchSize; b++)
          for (let j = 0; j < this.hiddenSize; j++)
            dhFromOutput.set(b, j, dOutput.get(b, t * this.hiddenSize + j));
        dh = dhFromOutput.add(dhNext);
      } else {
        dh = t === this.seqLength - 1 ? dOutput.add(dhNext) : dhNext;
      }

      const zt = cache.gates_z[t];
      const rt = cache.gates_r[t];
      const hCandidate = cache.candidates[t];
      const hPrev = cache.hiddens[t];
      const combined = cache.combined[t];
      const combinedReset = cache.combinedReset[t];

      // dh_t = (1-z_t) ⊙ dh → dh_prev, z_t ⊙ dh → dh_candidate
      // dz = (h_candidate - h_prev) ⊙ dh ⊙ σ'(z)
      const dzRaw = new Matrix(batchSize, this.hiddenSize);
      const dhCandidate = new Matrix(batchSize, this.hiddenSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          const z = zt.get(b, j);
          dhCandidate.set(b, j, dh.get(b, j) * z);
          dzRaw.set(b, j, dh.get(b, j) * (hCandidate.get(b, j) - hPrev.get(b, j)));
        }
      }

      const dzt = dzRaw.mul(this._sigmoidGrad(zt));
      const dhCandRaw = dhCandidate.mul(this._tanhGrad(hCandidate));

      // Gradients for Wh through combinedReset
      dWh = dWh.add(combinedReset.T().dot(dhCandRaw));
      for (let j = 0; j < this.hiddenSize; j++) {
        let s = 0;
        for (let b = 0; b < batchSize; b++) s += dhCandRaw.get(b, j);
        dbh.set(0, j, dbh.get(0, j) + s);
      }

      // dCombinedReset
      const dCombinedReset = dhCandRaw.dot(this.Wh.T());

      // dr from combinedReset: r affects only the h_prev part
      const drt = new Matrix(batchSize, this.hiddenSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          drt.set(b, j, dCombinedReset.get(b, this.inputSize + j) * hPrev.get(b, j));
        }
      }
      const drtGated = drt.mul(this._sigmoidGrad(rt));

      // Gradients for Wz, Wr
      dWz = dWz.add(combined.T().dot(dzt));
      dWr = dWr.add(combined.T().dot(drtGated));
      for (let j = 0; j < this.hiddenSize; j++) {
        let sz = 0, sr = 0;
        for (let b = 0; b < batchSize; b++) { sz += dzt.get(b, j); sr += drtGated.get(b, j); }
        dbz.set(0, j, dbz.get(0, j) + sz);
        dbr.set(0, j, dbr.get(0, j) + sr);
      }

      // dCombined from z and r gates
      const dCombined = dzt.dot(this.Wz.T()).add(drtGated.dot(this.Wr.T()));

      // Input gradient
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.inputSize; j++) {
          dInput.set(b, t * this.inputSize + j,
            dCombined.get(b, j) + dCombinedReset.get(b, j));
        }
      }

      // dhNext = (1-z) ⊙ dh + gradient from combined + reset contributions
      dhNext = new Matrix(batchSize, this.hiddenSize);
      for (let b = 0; b < batchSize; b++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dhNext.set(b, j,
            dh.get(b, j) * (1 - zt.get(b, j)) +
            dCombined.get(b, this.inputSize + j) +
            dCombinedReset.get(b, this.inputSize + j) * rt.get(b, j));
        }
      }
    }

    this._dWz = dWz; this._dWr = dWr; this._dWh = dWh;
    this._dbz = dbz; this._dbr = dbr; this._dbh = dbh;
    this.dWeights = dWz;
    this.dBiases = dbz;

    return dInput;
  }

  update(learningRate) {
    const scale = learningRate / (this._cache ? this._cache.inputs[0].rows : 1);
    this.Wz = this.Wz.sub(this._dWz.mul(scale));
    this.Wr = this.Wr.sub(this._dWr.mul(scale));
    this.Wh = this.Wh.sub(this._dWh.mul(scale));
    this.bz = this.bz.sub(this._dbz.mul(scale));
    this.br = this.br.sub(this._dbr.mul(scale));
    this.bh = this.bh.sub(this._dbh.mul(scale));
  }

  paramCount() {
    const combined = this.inputSize + this.hiddenSize;
    return 3 * (combined * this.hiddenSize + this.hiddenSize); // 3 gates
  }
}

// causal-conv1d.js — Causal 1D Convolution (used in Mamba, WaveNet)
export function causalConv1d(input, kernel) {
  const seqLen = input.length;
  const kLen = kernel.length;
  const output = new Float64Array(seqLen);
  
  for (let i = 0; i < seqLen; i++) {
    let sum = 0;
    for (let k = 0; k < kLen; k++) {
      const idx = i - k;
      if (idx >= 0) sum += input[idx] * kernel[k];
    }
    output[i] = sum;
  }
  return output;
}

// Depthwise separable causal conv
export function depthwiseCausalConv(channels, kernel) {
  return channels.map(ch => causalConv1d(ch, kernel));
}

// swiglu.js — SwiGLU FFN re-export from modern-decoder
// SwiGLU (Shazeer 2020): gate = SiLU(x * W_gate), output = gate * (x * W_up) * W_down
// 2/3 the params of standard FFN at same performance. Used in Llama, Mistral, PaLM.

export { SwiGLUFFN } from './modern-decoder.js';

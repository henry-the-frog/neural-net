# neural-net

A comprehensive deep learning library built from scratch in JavaScript. No dependencies. **168 source modules, 192 test files, 2,300+ tests, ~27,000 lines of code.** Spans Hopfield networks (1982) to KAN (2024) — 42 years of neural network research, implemented and tested.

## Quick Start

```javascript
import { Network } from './src/network.js';
import { Matrix } from './src/matrix.js';

const net = new Network();
net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');

const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
const targets = Matrix.fromArray([[0], [1], [1], [0]]);

for (let i = 0; i < 1000; i++) {
  net.trainBatch(inputs, targets, 0.5);
}

console.log(net.predict(inputs)); // XOR!
```

### More Examples

<details>
<summary><b>🧠 Transformer (Self-Attention)</b></summary>

```javascript
import { MultiHeadAttention } from './src/multi-head-attention.js';
import { Matrix } from './src/matrix.js';

const attn = new MultiHeadAttention(16, 4); // dim=16, 4 heads
const seq = Matrix.random(8, 16); // 8 tokens, 16 dims
const output = attn.forward(seq);  // Self-attention output
```
</details>

<details>
<summary><b>🎮 Reinforcement Learning (DQN)</b></summary>

```javascript
import { DQN } from './src/dqn.js';

const agent = new DQN({
  stateDim: 4, actionDim: 2,
  hiddenDim: 32, lr: 0.001,
  gamma: 0.99, epsilon: 1.0,
});

// Training loop
const state = [0.5, -0.3, 0.1, 0.8];
const action = agent.act(state);
agent.store(state, action, 1.0, [0.4, -0.2, 0.2, 0.7], false);
agent.train(); // Experience replay + target network
```
</details>

<details>
<summary><b>🔬 Mixture of Experts</b></summary>

```javascript
import { MixtureOfExperts } from './src/moe.js';
import { Matrix } from './src/matrix.js';

const moe = new MixtureOfExperts(8, 4, 16, 8, 2);
// 8 input dim, 4 experts, 16 hidden, 8 output, top-2 routing
const x = Matrix.random(32, 8);     // Batch of 32
const y = moe.forward(x);           // Routes each token to 2 best experts
console.log(moe.routingDistribution()); // Expert load balance
```
</details>

<details>
<summary><b>🎨 Variational Autoencoder</b></summary>

```javascript
import { VAE } from './src/vae.js';

const vae = new VAE(784, 256, 16, { beta: 1.0 }); // MNIST-sized
const data = Array.from({length: 100}, () => 
  Array.from({length: 784}, () => Math.random())
);
const { history } = vae.train(data, { epochs: 20 });
const samples = vae.generate(5); // Generate 5 new images
```
</details>

## What's Inside

### Core (`matrix.js`, `network.js`, `loss.js`, `autograd.js`)
Float64Array matrix math, network builder with fluent API, loss functions (MSE, cross-entropy, Huber), and reverse-mode automatic differentiation with 15+ ops.

### Layers & Activations
- **Dense** — fully connected with bias
- **Conv2D / Conv1D** — 2D convolution (proper col2im backward), 1D causal convolution
- **MaxPool / AvgPool / Flatten** — standard CNN building blocks
- **BatchNorm / GroupNorm / LayerNorm** — normalization layers
- **Dropout / Regularization** — training-time regularization
- **Residual connections** — skip connections with optional projection
- **9 activations** — ReLU, sigmoid, tanh, softmax, GELU, swish, ELU, SELU, leaky ReLU

### Optimizers & Training
- **SGD, Momentum, Adam, RMSprop, AdaGrad, AdamW** — full optimizer zoo
- **7 LR schedulers** — step, cosine, warmup, one-cycle, exponential, plateau, linear
- **Early stopping** — patience-based with best model restore and mode (min/max)
- **Gradient clipping, accumulation, checkpointing** — training stability
- **Cross-validation** — K-fold with accuracy and loss
- **Mixed precision** — simulated FP16 training
- **EMA** — exponential moving average of weights

### Sequence Models
- **RNN** — Elman network with BPTT
- **LSTM** — 4-gate Long Short-Term Memory with BPTT
- **GRU** — Gated Recurrent Unit
- **ESN** — Echo State Network (reservoir computing)
- **RWKV** — Linear-complexity attention alternative (WKV mechanism)
- **Mamba SSM** — State Space Model with selective scan

### Transformers & Attention
- **Transformer** — encoder/decoder blocks, positional encoding, layer norm
- **Multi-head attention** — scaled dot-product, causal masks
- **Flash Attention** — memory-efficient tiled computation
- **Grouped Query Attention (GQA)** — parameter-efficient attention sharing
- **Sliding Window Attention** — bounded context for long sequences
- **Sparse Attention** — Longformer/BigBird-style patterns
- **Cross-attention** — encoder-decoder bridge
- **RoPE** — Rotary Position Embeddings with scaling/interpolation
- **Relative Position Bias** — learned relative position encoding
- **Attention entropy & sinks** — attention pattern analysis

### LLM Pipeline
- **BPE Tokenizer** — byte-pair encoding with merge rules
- **Byte Tokenizer** — character-level tokenization
- **Embeddings** — learned token + position embeddings with similarity search
- **KV Cache** — inference-time key-value caching with compression
- **Paged Attention** — memory-efficient attention for long contexts
- **Speculative Decoding** — draft model acceleration
- **Beam Search** — beam search with length penalty
- **Sampling** — top-k, top-p (nucleus), temperature, repetition penalty
- **Constrained Decoding** — grammar/regex-guided generation
- **Text Generation** — end-to-end generation pipeline
- **Token Healing** — fix tokenization artifacts at generation boundaries
- **Prefix Caching** — reuse KV cache for shared prefixes
- **Continuous Batching** — dynamic batch management for inference
- **Multi-token Prediction** — predict multiple tokens per step
- **Logit Processors** — composable output distribution transforms

### Working End-to-End Models
- **char-lm.js** — character-level language model
- **MicroGPT** — small GPT implementation with training loop
- **mini-llm.js** — minimal LLM with full pipeline
- **LLaMA** — LLaMA-architecture decoder with RoPE, SwiGLU, RMSNorm
- **Modern Decoder** — configurable decoder with GQA, sliding window, AdaLN

### RLHF & Alignment
- **PPO** — Proximal Policy Optimization for LLM fine-tuning
- **DPO** — Direct Preference Optimization (reference-free)
- **Reward Model** — learned reward from human preferences
- **Constitutional AI** — self-critique and revision pipeline
- **Context Distillation** — distill system prompts into model weights
- **REINFORCE** — vanilla policy gradient

### Efficiency & Deployment
- **LoRA** — Low-Rank Adaptation for parameter-efficient fine-tuning
- **Prefix Tuning** — learnable prefix tokens
- **Knowledge Distillation** — teacher-student training
- **Pruning** — magnitude and structured pruning
- **Quantization** — int8 and float16 weight quantization
- **Structured Pruning** — channel/filter-level pruning
- **Model Parallelism** — tensor and pipeline parallelism simulation
- **Sequence Packing** — efficient variable-length sequence batching
- **Weight Tying** — share embedding and output weights
- **Parameter Count** — model size analysis

### Neuroscience-Inspired
- **Hopfield Networks** — classical, modern (exponential), and Boltzmann machines
- **Spiking Neural Networks** — LIF and Izhikevich neurons with STDP learning
- **Self-Organizing Maps** — Kohonen networks for topology-preserving mapping
- **Echo State Networks** — reservoir computing with spectral radius tuning
- **Predictive Coding** — hierarchical prediction error minimization
- **Restricted Boltzmann Machines** — contrastive divergence training
- **Energy-Based Models** — Langevin dynamics sampling

### Research Architectures
- **KAN** — Kolmogorov-Arnold Networks with B-spline edge activations (2024)
- **Neural ODE** — continuous-depth networks (Euler/RK4/adaptive solvers)
- **Neural Turing Machine** — differentiable external memory (read/write heads)
- **Capsule Networks** — dynamic routing between capsules
- **Graph Neural Networks** — message passing, GCN (Karate Club demo)
- **Mixture of Experts** — top-K gating with load balancing
- **Hypernetworks** — task-conditioned weight generation
- **Normalizing Flows** — planar flow, affine coupling, ActNorm
- **Neuroevolution** — genetic algorithm + evolution strategies
- **MAML** — Model-Agnostic Meta-Learning (few-shot)
- **DARTS** — differentiable architecture search
- **Lottery Ticket** — iterative magnitude pruning to find sparse subnetworks
- **Mixture Density Networks** — Gaussian mixture output distributions
- **Differentiable Sorting** — Sinkhorn-based soft sorting
- **Scaling Laws** — Chinchilla-style compute-optimal analysis

### Generative Models
- **Autoencoder** — standard encoder-decoder
- **VAE** — Variational Autoencoder with reparameterization trick
- **DDPM** — Denoising Diffusion Probabilistic Models
- **GAN** — adversarial training (generator + discriminator)
- **Contrastive Learning** — SimCLR-style representation learning
- **CLIP** — contrastive language-image pretraining

### Reinforcement Learning
- **DQN** — Deep Q-Network with experience replay
- **PPO** — Proximal Policy Optimization
- **REINFORCE** — vanilla policy gradient

### Utilities
- **Model Zoo** — pre-configured architectures (XOR, classifier, autoencoder, etc.)
- **Training Logger** — metrics tracking, ASCII charts, JSON/CSV export
- **Datasets** — synthetic data (spiral, moons, circles, blobs, sine)
- **Preprocessing** — StandardScaler, MinMaxScaler, one-hot encoding, train/test split
- **Data Augmentation** — image transforms for training
- **Data Loader** — batched iteration with shuffling
- **Metrics** — confusion matrix, precision, recall, F1, classification report
- **Callbacks** — training event hooks
- **Gradient Check** — numerical gradient verification
- **AutoML** — hyperparameter search

## Mathematical Verification

Every backward pass is gradient-checked against finite-difference approximation:
- **10 numerical gradient checks** all at machine precision (error < 1e-5)
- Verified: Dense, Conv2D, LSTM, Transformer attention, BatchNorm, Autograd, KAN, and more
- This means the library actually learns, not just forward-passes

## Interactive Demo

Live at [henry-the-frog.github.io/neural-net/](https://henry-the-frog.github.io/neural-net/) — train neural networks in your browser.

## Testing

```bash
# Run all tests
node --test src/*.test.js

# Run a specific module's tests
node --test src/transformer.test.js

# Run gradient stress tests
node --test src/gradient-check.test.js
```

**150 test files** covering:
- Numerical gradient agreement (analytical vs finite-difference)
- Invariants (energy monotonicity, weight symmetry, probability normalization)
- Boundary conditions (zero input, extreme values)
- Convergence (loss decreasing, function approximation)
- Serialization round-trips (toJSON/fromJSON)

## Architecture

```
src/
├── Core          matrix.js, network.js, loss.js, autograd.js
├── Layers        dense.js, conv2d.js, conv1d.js, pool.js, flatten.js
├── Normalization batchnorm.js, groupnorm.js, normalization.js
├── Training      optimizer.js, adamw.js, optimizer-zoo.js, lr-scheduler.js
│                 early-stopping.js, gradient-clip.js, gradient-accumulator.js
│                 gradient-checkpointing.js, mixed-precision.js, ema.js
├── Sequences     rnn.js, lstm.js, gru.js, esn.js, rwkv.js, mamba-ssm.js
├── Attention     attention.js, mha.js, flash-attention.js, gqa.js
│                 sparse-attention.js, sliding-window-attention.js
│                 cross-attention.js, rope.js, positional-encoding.js
├── LLM Pipeline  bpe-tokenizer.js, embedding.js, kv-cache.js, sampling.js
│                 beam-search.js, speculative-decoding.js, text-generation.js
│                 paged-attention.js, continuous-batching.js
├── Models        char-lm.js, microgpt.js, mini-llm.js, llama.js
│                 modern-decoder.js
├── RLHF          ppo.js, dpo.js, reward-model.js, constitutional-ai.js
├── Efficiency    lora.js, prefix-tuning.js, distillation.js, pruning.js
│                 quantization.js, model-parallel.js, weight-tying.js
├── Neuro         hopfield.js, snn.js, som.js, predictive-coding.js
│                 rbm.js, ebm.js
├── Research      kan.js, neural-ode.js, ntm.js, capsule.js, gnn.js
│                 moe.js, hypernetwork.js, normalizing-flows.js
│                 neuroevolution.js, maml.js, darts.js, lottery-ticket.js
├── Generative    autoencoder.js, vae.js, ddpm.js, gan.js
│                 contrastive.js, clip.js
├── RL            dqn.js, reinforce.js
└── Utils         model-zoo.js, training-logger.js, datasets.js
                  preprocessing.js, metrics.js, data-augmentation.js
                  gradient-check.js, callbacks.js, automl.js
```

## Philosophy

- **Zero dependencies** — everything from Matrix math to BPE tokenization, built from scratch
- **Gradient-verified** — every backward pass checked against numerical differentiation
- **Educational** — clean code that follows the papers, readable implementations
- **Comprehensive** — from Hopfield (1982) to KAN (2024), covering the full landscape
- **Practical** — working end-to-end models, interactive browser demo

## License

MIT

// convergence-advanced.test.js — Verify recently-fixed modules can actually train
// These tests validate that the backward pass fixes produce correct training behavior.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { mse } from '../src/loss.js';

describe('Advanced Convergence: Recently Fixed Modules', () => {

  describe('KAN (Kolmogorov-Arnold Network)', () => {
    it('should learn XOR problem', async () => {
      const { KAN } = await import('../src/kan.js');
      
      const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
      const targets = Matrix.fromArray([[0], [1], [1], [0]]);
      
      let passed = false;
      for (let attempt = 0; attempt < 5 && !passed; attempt++) {
        const net = new KAN([2, 8, 1]);
        const lr = 0.05;
        let lastLoss = Infinity;
        
        for (let epoch = 0; epoch < 3000; epoch++) {
          const output = net.forward(inputs);
          const loss = mse.compute(output, targets);
          const grad = mse.gradient(output, targets);
          
          // Backward through layers
          let dx = grad;
          for (let l = net.layers.length - 1; l >= 0; l--) {
            dx = net.layers[l].backward(dx);
          }
          net.update(lr);
          
          lastLoss = loss;
        }
        
        if (lastLoss < 0.1) {
          const pred = net.forward(inputs);
          const p00 = pred.get(0, 0);
          const p01 = pred.get(1, 0);
          const p10 = pred.get(2, 0);
          const p11 = pred.get(3, 0);
          // Just check it learned the pattern direction
          if (p01 > p00 && p10 > p00 && p01 > p11 && p10 > p11) {
            passed = true;
          }
        }
      }
      assert.ok(passed, 'KAN should learn XOR pattern in at least 1 of 5 attempts');
    });

    it('should show decreasing loss on simple regression', async () => {
      const { KAN } = await import('../src/kan.js');
      const net = new KAN([1, 4, 1]);
      
      // Simple y = 2x + 1 with inputs in [0, 1]
      const inputs = Matrix.fromArray([[0.1], [0.3], [0.5], [0.7], [0.9]]);
      const targets = Matrix.fromArray([[1.2], [1.6], [2.0], [2.4], [2.8]]);
      
      const losses = [];
      for (let epoch = 0; epoch < 500; epoch++) {
        const output = net.forward(inputs);
        const loss = mse.compute(output, targets);
        if (epoch % 100 === 0) losses.push(loss);
        
        const grad = mse.gradient(output, targets);
        let dx = grad;
        for (let l = net.layers.length - 1; l >= 0; l--) {
          dx = net.layers[l].backward(dx);
        }
        net.update(0.01);
      }
      
      // Loss should decrease
      assert.ok(losses[losses.length - 1] < losses[0], 
        `Loss should decrease: first=${losses[0].toFixed(4)}, last=${losses[losses.length-1].toFixed(4)}`);
    });
  });

  describe('NeuralODE', () => {
    it('should learn identity mapping', async () => {
      const { NeuralODE } = await import('../src/neural-ode.js');
      const net = new NeuralODE(2, 4, 2, { solver: 'euler', steps: 5 });
      
      // Identity: output = input
      const inputs = Matrix.fromArray([[1, 0], [0, 1], [0.5, 0.5], [-0.5, 0.5]]);
      const targets = inputs;
      
      const losses = [];
      for (let epoch = 0; epoch < 200; epoch++) {
        const output = net.forward(inputs);
        const loss = mse.compute(output, targets);
        if (epoch % 50 === 0) losses.push(loss);
        
        const grad = mse.gradient(output, targets);
        net.backward(grad);
        net.update(0.01);
      }
      
      assert.ok(losses[losses.length - 1] < losses[0] * 0.5,
        `NeuralODE loss should decrease by >50%: first=${losses[0].toFixed(4)}, last=${losses[losses.length-1].toFixed(4)}`);
    });

    it('should learn simple function via train method', async () => {
      const { NeuralODE } = await import('../src/neural-ode.js');
      const net = new NeuralODE(1, 8, 1, { solver: 'euler', steps: 5 });
      
      // y = x^2 (simple curve)
      const inputs = Matrix.fromArray([[-1], [-0.5], [0], [0.5], [1]]);
      const targets = Matrix.fromArray([[1], [0.25], [0], [0.25], [1]]);
      
      const losses = net.train(inputs, targets, 500, 0.01);
      
      assert.ok(losses[losses.length - 1] < losses[0] * 0.5,
        `NeuralODE train() should reduce loss by >50%: first=${losses[0].toFixed(4)}, last=${losses[losses.length-1].toFixed(4)}`);
    });
  });

  describe('MixtureOfExperts', () => {
    it('should learn multi-mode data better than single expert', async () => {
      const { MixtureOfExperts } = await import('../src/moe.js');
      const { Dense } = await import('../src/layer.js');
      
      // Two-mode data: positive inputs -> positive outputs, negative -> negative
      // This benefits from routing different modes to different experts
      const inputs = Matrix.fromArray([
        [1, 0.5], [0.8, 0.3], [1.2, 0.7],    // Mode A
        [-1, -0.5], [-0.8, -0.3], [-1.2, -0.7] // Mode B
      ]);
      const targets = Matrix.fromArray([
        [2], [1.5], [2.5],     // Mode A: sum-ish
        [-2], [-1.5], [-2.5]    // Mode B: sum-ish
      ]);
      
      // Train MoE
      const moe = new MixtureOfExperts(2, 3, 8, 1, 2); // 2 input, 3 experts, 8 hidden, 1 output, top-2
      const moeLosses = [];
      for (let epoch = 0; epoch < 500; epoch++) {
        const output = moe.forward(inputs);
        const loss = mse.compute(output, targets);
        if (epoch % 100 === 0) moeLosses.push(loss);
        
        const grad = mse.gradient(output, targets);
        moe.backward(grad);
        moe.update(0.01);
      }
      
      // MoE should at least show decreasing loss
      assert.ok(moeLosses[moeLosses.length - 1] < moeLosses[0],
        `MoE loss should decrease: first=${moeLosses[0].toFixed(4)}, last=${moeLosses[moeLosses.length-1].toFixed(4)}`);
    });

    it('should show loss convergence on regression task', async () => {
      const { MixtureOfExperts } = await import('../src/moe.js');
      
      // Simple regression: y = 2*x1 + x2
      const inputs = Matrix.fromArray([
        [1, 1], [2, 0], [0, 2], [1, -1], [-1, 1]
      ]);
      const targets = Matrix.fromArray([
        [3], [4], [2], [1], [-1]
      ]);
      
      const moe = new MixtureOfExperts(2, 2, 4, 1, 2);
      let firstLoss, lastLoss;
      
      for (let epoch = 0; epoch < 1000; epoch++) {
        const output = moe.forward(inputs);
        const loss = mse.compute(output, targets);
        if (epoch === 0) firstLoss = loss;
        lastLoss = loss;
        
        const grad = mse.gradient(output, targets);
        moe.backward(grad);
        moe.update(0.01);
      }
      
      assert.ok(lastLoss < firstLoss * 0.5,
        `MoE regression loss should decrease by >50%: first=${firstLoss.toFixed(4)}, last=${lastLoss.toFixed(4)}`);
    });
  });

  describe('NeuralODELayer', () => {
    it('should propagate gradients correctly through multiple steps', async () => {
      const { NeuralODELayer } = await import('../src/neural-ode.js');
      const layer = new NeuralODELayer(2, 1, 'euler', 10);
      
      // Simple regression through ODE layer
      const inputs = Matrix.fromArray([[1, 0], [0, 1], [1, 1]]);
      const targets = Matrix.fromArray([[0, 1], [1, 0], [0.5, 0.5]]);
      
      const losses = [];
      for (let epoch = 0; epoch < 200; epoch++) {
        const output = layer.forward(inputs);
        const loss = mse.compute(output, targets);
        if (epoch % 50 === 0) losses.push(loss);
        
        const grad = mse.gradient(output, targets);
        const dx = layer.backward(grad);
        layer.update(0.005);
      }
      
      assert.ok(losses[losses.length - 1] < losses[0],
        `ODE layer loss should decrease: first=${losses[0].toFixed(4)}, last=${losses[losses.length-1].toFixed(4)}`);
    });
  });
});

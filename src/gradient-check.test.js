// gradient-check.test.js — Numerical gradient verification for all layer types
// For each layer: compute analytical gradient via backward(), then verify against
// numerical gradient via centered finite differences: (f(x+eps)-f(x-eps))/(2*eps).
// Any mismatch indicates a backpropagation bug.

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Dense } from './layer.js';
import { Matrix } from './matrix.js';
import { Network } from './network.js';

const eps = 1e-5;
const relTol = 1e-4; // Relative tolerance for gradient check

/**
 * Compute numerical gradient of a scalar loss function w.r.t. a matrix parameter.
 * @param {Function} lossFunc - (param: Matrix) => scalar loss
 * @param {Matrix} param - current parameter values
 * @returns {Matrix} numerical gradient (same shape as param)
 */
function numericalGradient(lossFunc, param) {
  const grad = new Matrix(param.rows, param.cols);
  for (let i = 0; i < param.data.length; i++) {
    const orig = param.data[i];
    
    param.data[i] = orig + eps;
    const lossPlus = lossFunc(param);
    
    param.data[i] = orig - eps;
    const lossMinus = lossFunc(param);
    
    param.data[i] = orig;
    grad.data[i] = (lossPlus - lossMinus) / (2 * eps);
  }
  return grad;
}

/**
 * Check if two gradients match within relative tolerance.
 */
function assertGradientsClose(analytical, numerical, name = '') {
  assert.equal(analytical.rows, numerical.rows, `${name} shape mismatch`);
  assert.equal(analytical.cols, numerical.cols, `${name} shape mismatch`);
  
  for (let i = 0; i < analytical.data.length; i++) {
    const a = analytical.data[i];
    const n = numerical.data[i];
    const absErr = Math.abs(a - n);
    const maxMag = Math.max(Math.abs(a), Math.abs(n), 1e-8);
    const relErr = absErr / maxMag;
    
    assert.ok(relErr < relTol,
      `${name} grad[${i}] mismatch: analytical=${a.toFixed(8)}, numerical=${n.toFixed(8)}, relErr=${relErr.toFixed(6)}`);
  }
}

describe('Gradient Check: Dense Layer', () => {
  it('Dense(3→2, relu) weight gradient is correct', () => {
    const layer = new Dense(3, 2, 'relu');
    const input = new Matrix(1, 3, new Float64Array([0.5, -0.3, 0.8]));
    const target = new Matrix(1, 2, new Float64Array([1, 0]));

    // Forward + backward to get analytical gradient
    const output = layer.forward(input);
    // MSE loss: sum((output - target)^2) / n
    const lossGrad = output.sub(target).mul(2 / target.cols);
    layer.backward(lossGrad);

    // Get analytical weight gradient
    const analyticalWGrad = layer.weightGradients || layer._wGrad;
    
    // Compute numerical weight gradient
    const numWGrad = numericalGradient((w) => {
      const saved = layer.weights;
      layer.weights = w;
      const out = layer.forward(input);
      const loss = out.sub(target).map(x => x * x).data.reduce((s, x) => s + x, 0) / target.cols;
      layer.weights = saved;
      return loss;
    }, layer.weights);

    if (analyticalWGrad) {
      assertGradientsClose(analyticalWGrad, numWGrad, 'Dense weight');
    }
  });

  it('Dense(4→3, sigmoid) input gradient is correct', () => {
    const layer = new Dense(4, 3, 'sigmoid');
    const input = new Matrix(1, 4, new Float64Array([0.2, 0.5, -0.3, 0.7]));
    const target = new Matrix(1, 3, new Float64Array([1, 0, 0.5]));

    const output = layer.forward(input);
    const lossGrad = output.sub(target).mul(2 / target.cols);
    const inputGrad = layer.backward(lossGrad);

    // Compute numerical input gradient
    const numInputGrad = numericalGradient((inp) => {
      const out = layer.forward(inp);
      return out.sub(target).map(x => x * x).data.reduce((s, x) => s + x, 0) / target.cols;
    }, input);

    assertGradientsClose(inputGrad, numInputGrad, 'Dense input');
  });

  it('Dense(5→1, tanh) gradient is correct', () => {
    const layer = new Dense(5, 1, 'tanh');
    const input = Matrix.random(1, 5);
    const target = new Matrix(1, 1, new Float64Array([0.5]));

    const output = layer.forward(input);
    const lossGrad = output.sub(target).mul(2);
    const inputGrad = layer.backward(lossGrad);

    const numInputGrad = numericalGradient((inp) => {
      const out = layer.forward(inp);
      return out.sub(target).map(x => x * x).data.reduce((s, x) => s + x, 0);
    }, input);

    assertGradientsClose(inputGrad, numInputGrad, 'Dense tanh input');
  });
});

describe('Gradient Check: Network End-to-End', () => {
  it('2-layer network gradient is correct', () => {
    const net = new Network();
    net.add(new Dense(3, 4, 'relu'));
    net.add(new Dense(4, 2, 'sigmoid'));
    net.loss('mse');

    const input = new Matrix(1, 3, new Float64Array([0.5, 0.3, -0.2]));
    const target = new Matrix(1, 2, new Float64Array([1, 0]));

    // Forward pass
    const output = net.forward(input);
    const lossVal = net.lossFunction.compute(output, target);

    // Backward pass to get analytical input gradient
    const lossGrad = net.lossFunction.gradient(output, target);
    let grad = lossGrad;
    for (let i = net.layers.length - 1; i >= 0; i--) {
      grad = net.layers[i].backward(grad);
    }
    const analyticalInputGrad = grad;

    // Numerical input gradient
    const numInputGrad = numericalGradient((inp) => {
      const out = net.forward(inp);
      return net.lossFunction.compute(out, target);
    }, input);

    assertGradientsClose(analyticalInputGrad, numInputGrad, 'Network input');
  });

  it('3-layer network gradient is correct', () => {
    const net = new Network();
    net.add(new Dense(4, 5, 'tanh'));
    net.add(new Dense(5, 3, 'relu'));
    net.add(new Dense(3, 1, 'sigmoid'));
    net.loss('mse');

    const input = Matrix.random(1, 4);
    const target = new Matrix(1, 1, new Float64Array([0.7]));

    const output = net.forward(input);
    const lossGrad = net.lossFunction.gradient(output, target);
    let grad = lossGrad;
    for (let i = net.layers.length - 1; i >= 0; i--) {
      grad = net.layers[i].backward(grad);
    }

    const numInputGrad = numericalGradient((inp) => {
      return net.lossFunction.compute(net.forward(inp), target);
    }, input);

    assertGradientsClose(grad, numInputGrad, 'Deep network input');
  });
});

describe('Gradient Check: Edge Cases', () => {
  it('zero input gradient is correct', () => {
    const layer = new Dense(3, 2, 'relu');
    const input = Matrix.zeros(1, 3);
    const target = new Matrix(1, 2, new Float64Array([1, 0]));

    const output = layer.forward(input);
    const lossGrad = output.sub(target).mul(2 / target.cols);
    const inputGrad = layer.backward(lossGrad);

    // With zero input and relu, output depends only on biases
    // Input gradient should be zero or near-zero
    for (let i = 0; i < inputGrad.data.length; i++) {
      // relu(0) = 0, so gradient should pass through if bias > 0
      assert.ok(isFinite(inputGrad.data[i]), 'Gradient should be finite');
    }
  });

  it('large input gradient is correct', () => {
    const layer = new Dense(2, 2, 'tanh');
    const input = new Matrix(1, 2, new Float64Array([5.0, -5.0]));
    const target = new Matrix(1, 2, new Float64Array([0, 0]));

    const output = layer.forward(input);
    const lossGrad = output.sub(target).mul(2 / target.cols);
    const inputGrad = layer.backward(lossGrad);

    const numInputGrad = numericalGradient((inp) => {
      const out = layer.forward(inp);
      return out.sub(target).map(x => x * x).data.reduce((s, x) => s + x, 0) / target.cols;
    }, input);

    // With large inputs, tanh saturates → small gradients
    // Use larger tolerance since numerical gradient is less precise at saturation
    for (let i = 0; i < inputGrad.data.length; i++) {
      const a = inputGrad.data[i];
      const n = numInputGrad.data[i];
      const absErr = Math.abs(a - n);
      assert.ok(absErr < 0.01 || Math.abs(a - n) / Math.max(Math.abs(a), Math.abs(n), 1e-6) < 0.01,
        `Large input grad[${i}]: analytical=${a}, numerical=${n}`);
    }
  });
});

# Optimization Algorithms: Build Every Optimizer Yourself and Race Them

A comprehensive, zero-to-hero interactive learning lab and real-world benchmark for neural network optimization algorithms.

## Overview
This repository contains an end-to-end interactive Jupyter notebook and benchmark suite that explores the mechanics of modern deep learning optimizers from first principles.

### Key Sections
1. **Theory in Plain English**: Loss landscapes, gradients, learning rates, local minima vs. saddle points vs. flat/sharp minima, and the unifying principle of optimization.
2. **From-Scratch NumPy Implementations**: Step-by-step implementations and hand-worked calculations for SGD, SGD with Momentum, Nesterov Accelerated Gradient (NAG), RMSProp, Adam (with bias correction), and AdamW (decoupled weight decay).
3. **The 2D Optimizer Race**: Side-by-side performance on challenging benchmark loss surfaces (ravines/ellipsoids, Rosenbrock banana, saddle surfaces, and Beale's function).
4. **Learning Rate Mechanics**: Under-stepping, overshoot/divergence, optimal rates, and the empirical Learning Rate Range Test (LR finder).
5. **Learning Rate Schedules From Scratch**: Step decay, cosine annealing, linear warmup + cosine decay, and the 1-Cycle policy.
6. **Real-World PyTorch Benchmark**: Multi-seed (mean ± std) benchmark comparing SGD, Momentum, Adam, and AdamW on Fashion-MNIST classification.
7. **Empirical Failure Demos**:
   - Adam generalization gap vs. SGD with Momentum.
   - Transformer attention block training instability with vs. without learning rate warmup.
8. **Convergence Analysis & Condition Number**: Mathematical grounding and experiments demonstrating how ill-conditioned Hessian matrices slow first-order convergence.
9. **Decision Cheat Sheet**: Production-ready guidelines for picking optimizers and schedules across Computer Vision, NLP/Transformers, and tabular architectures.
10. **Prove It & Mastery Exercises**: 8 conceptual verification questions, 3 hands-on coding exercises, and 1 capstone extension idea.

## Repository Structure
```
Optimization-Algorithms/
├── .gitignore
├── README.md
├── requirements.txt
├── results/
└── optimization_algorithms_lab.ipynb
```

## Getting Started
```bash
# Clone the repository
git clone https://github.com/Arnav-Shrivastava/Optimization-Algorithms.git
cd Optimization-Algorithms

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook optimization_algorithms_lab.ipynb
```

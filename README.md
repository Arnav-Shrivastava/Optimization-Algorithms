# 🏔️ Optimization Algorithms: Build Every Optimizer Yourself and Race Them

An end-to-end, zero-to-hero interactive learning lab and real-world benchmark suite exploring deep learning optimization algorithms from first principles.

---

## 📖 Project Overview

Optimization algorithms are the physical engines driving modern artificial intelligence. Every time weights update in a multi-billion-parameter LLM or a vision classifier, an optimizer decides which direction to step and how far to leap.

This repository implements every major deep learning optimizer from scratch in pure NumPy, races them across treacherous 2D benchmark terrains, analyzes learning rate dynamics, and subjects them to an honest multi-seed empirical benchmark on Fashion-MNIST in PyTorch.

---

## 🧠 What I Learned & Core Insights

1. **The Unifying Principle of Optimization:**
   Every modern optimizer is constructed from the exact same fundamental equation:
   $$\text{Parameter Update} = -\alpha \cdot \Big[ \text{Per-Parameter Adaptive Scaling} \Big] \cdot \Big[ \text{Memory of Past Gradients} \Big]$$
   - **Plain SGD:** No memory, uniform scaling.
   - **Momentum & Nesterov (NAG):** Adds directional inertia via Exponential Moving Average (EMA) of gradients.
   - **RMSProp:** Adds coordinate-wise adaptive scaling by dividing by the root-mean-square of recent gradients.
   - **Adam:** Combines both Momentum (first moment) and RMSProp (second moment), plus bias correction.
   - **AdamW:** Decouples weight decay directly from gradient accumulation so regularization applies uniformly across all parameters.

2. **Condition Number ($\kappa$) Governs Convergence:**
   On quadratic loss surfaces $f(x) = \frac{1}{2} x^T A x - b^T x$, the rate of convergence is mathematically bound by the Hessian condition number $\kappa = \lambda_{\text{max}} / \lambda_{\text{min}}$:
   $$\|x_t - x^*\| \le \left( \frac{\kappa - 1}{\kappa + 1} \right)^t \|x_0 - x^*\|$$
   When $\kappa = 1$ (isotropic bowl), gradient descent converges in a single step. When $\kappa \gg 1$ (elongated ravines), plain gradient descent is crippled without momentum or adaptive preconditioners.

3. **The Adam Generalization Gap:**
   While Adam converges fastest during early epochs, well-tuned **SGD with Momentum** frequently reaches flatter minima that yield superior test accuracy on convolutional vision tasks ($+0.6\%$ to $+1.5\%$).

4. **Why Warmup is Mandatory for Transformers:**
   Without linear warmup, initial chaotic gradients from uncalibrated softmax attention layers corrupt Adam's second-moment buffer ($v_t$), permanently freezing attention weights at high loss.

---

## 🏁 The 2D Optimizer Race

All six from-scratch optimizers were launched across standard non-convex and ill-conditioned mathematical benchmark surfaces.

### 1. The Ravine ($f(x, y) = 0.5x^2 + 10y^2$)
*Curvature in $y$ is 20 times steeper than in $x$. Plain SGD oscillates violently between canyon walls, while adaptive optimizers (RMSProp, Adam, AdamW) equalize coordinate scales to glide straight down the valley floor.*

![Ravine Optimizer Race](results/surface_race_ravine.png)

### 2. The Rosenbrock "Banana" Function ($f(x, y) = (1-x)^2 + 100(y-x^2)^2$)
*A narrow parabolic canyon where the slope turns continuously. SGD gets stuck near the sharp bend, while Momentum and Adam navigate around the curve toward $(1, 1)$.*

![Rosenbrock Banana Race](results/surface_race_rosenbrock.png)

### 3. The Saddle Surface ($f(x, y) = x^2 - y^2$)
*A zero-gradient pass at $(0, 0)$. Momentum and NAG preserve velocity, escaping the flat saddle plateau and plunging down the negative slope.*

![Saddle Surface Race](results/surface_race_saddle.png)

### 4. Beale's Function
*A treacherous landscape with sharp corners and flat plains. Adam and AdamW reach the global minimum at $(3, 0.5)$ orders of magnitude faster than unadapted gradient descent.*

![Beale Function Race](results/surface_race_beale.png)

---

## 📊 Real-World Benchmark on Fashion-MNIST (PyTorch)

To ensure scientific honesty, a Convolutional Neural Network (CNN) was trained across **3 independent random seeds** ($42, 123, 999$). Below are the empirical results:

### Summary Benchmark Table (Mean $\pm$ Std)

| Optimizer | Learning Rate | Configuration | Final Test Accuracy (Mean $\pm$ Std) | Convergence Speed |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | $0.05$ | Base | **77.17% $\pm$ 0.57%** | Slow |
| **SGD + Momentum** | $0.02$ | $\beta = 0.9$ | **84.62% $\pm$ 0.71%** | Fast |
| **Adam** | $0.001$ | $\beta_1=0.9, \beta_2=0.999$ | **84.03% $\pm$ 0.66%** | Fastest Initial |
| **AdamW** | $0.001$ | $\lambda = 0.01$ | **83.89% $\pm$ 0.48%** | Fastest Initial |

![Fashion-MNIST Convergence & Generalization](results/fashion_mnist_convergence.png)

*Key Benchmark Finding:* Adam and AdamW descend training loss rapidly in Epoch 1, but **SGD with Momentum** achieves the highest final generalization accuracy on the unseen 10,000-image test set.

---

## ⚠️ Empirical Failure Demos

### Failure Demo A: The Adam Generalization Gap
On image classification, Adam's coordinate-wise normalization biases updates toward sharp minima that fit training batches aggressively but exhibit slightly lower test margin compared to SGD with Momentum.

### Failure Demo B: Transformer Instability Without Warmup
A 4-layer Transformer was trained on sequence classification with peak learning rate $\alpha = 0.02$:
- **Without Warmup:** Training loss immediately stalled at $\sim 3.84$ due to second-moment accumulator distortion.
- **With Linear Warmup (10 steps):** The model stabilized early attention representations and converged smoothly down to $2.51$.

![Transformer Warmup Failure Demo](results/failure_demo_warmup.png)

---

## 📐 Mathematical Convergence & Condition Number

We experimentally verified the theoretical convergence bound on quadratic surfaces with condition numbers $\kappa \in \{1, 10, 100\}$:

![Condition Number Convergence](results/condition_number_convergence.png)

*The empirical error trajectory strictly tracks the theoretical Polyak bound: $\|x_t - x^*\| \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^t \|x_0 - x^*\|$. When $\kappa = 100$, error contracts by barely $2\%$ per step.*

---

## 🧭 Practitioner's Decision Cheat Sheet

| Domain / Architecture | Recommended Optimizer | Default Base LR ($\alpha$) | Weight Decay ($\lambda$) | Recommended Schedule |
| :--- | :--- | :--- | :--- | :--- |
| **Vision (CNNs, ResNets)** | **SGD + Momentum** ($\beta = 0.9$) | $0.05 - 0.1$ | $10^{-4}$ (L2) | **Cosine Annealing** (or 1-Cycle) |
| **Vision Transformers (ViT)** | **AdamW** ($\beta_1 = 0.9, \beta_2 = 0.999$) | $5 \times 10^{-4} - 10^{-3}$ | $0.05$ (Decoupled) | **Linear Warmup + Cosine Decay** |
| **NLP & LLMs (Transformers)** | **AdamW** ($\beta_1 = 0.9, \beta_2 = 0.95-0.98$) | $10^{-4} - 5 \times 10^{-4}$ | $0.01 - 0.1$ | **Linear Warmup ($2-5\%$) + Cosine Decay** |
| **Tabular Neural Nets (MLPs)**| **AdamW** | $10^{-3} - 3 \times 10^{-3}$ | $10^{-4}$ | **ReduceLROnPlateau** (patience=5) |
| **Reinforcement Learning** | **Adam** | $3 \times 10^{-4}$ | $0.0$ | **Linear Decay** (or constant) |

---

## 📁 Repository Structure

```
Optimization-Algorithms/
├── .gitignore                          # Git exclusions (data/, checkpoints, cache)
├── README.md                           # Master project documentation & benchmarks
├── requirements.txt                    # Exact production dependencies
├── optimization_algorithms_lab.ipynb   # Complete, fully executed interactive notebook (166 cells)
└── results/                            # Benchmark artifacts and visualizations
    ├── condition_number_convergence.png
    ├── failure_demo_warmup.png
    ├── fashion_mnist_benchmark.md
    ├── fashion_mnist_convergence.png
    ├── surface_race_beale.png
    ├── surface_race_ravine.png
    ├── surface_race_rosenbrock.png
    └── surface_race_saddle.png
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Arnav-Shrivastava/Optimization-Algorithms.git
cd Optimization-Algorithms
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Interactive Notebook
```bash
jupyter notebook optimization_algorithms_lab.ipynb
```
*All cells are pre-executed with cached outputs, contour animations, and PyTorch training metrics.*

---

## 📜 License
MIT License. Created for learners, researchers, and engineers seeking first-principles intuition for neural network optimization.

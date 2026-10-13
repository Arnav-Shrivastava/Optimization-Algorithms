# Fashion-MNIST Benchmark Results (3 Seeds)

| Optimizer | Learning Rate | Other Config | Test Accuracy (Mean +/- Std) | Convergence Speed |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | 0.05 | None | **77.17% +/- 0.57%** | Slow |
| **SGD+Momentum** | 0.02 | beta=0.9 | **84.62% +/- 0.71%** | Fast |
| **Adam** | 0.001 | beta1=0.9, beta2=0.999 | **84.03% +/- 0.66%** | Fast |
| **AdamW** | 0.001 | decay=0.01 | **83.89% +/- 0.48%** | Fast |

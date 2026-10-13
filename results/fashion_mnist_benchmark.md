# Fashion-MNIST Benchmark Results (3 Seeds)

| Optimizer | Learning Rate | Other Config | Test Accuracy (Mean +/- Std) | Convergence Speed |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | 0.05 | None | **77.21% +/- 0.44%** | Slow |
| **SGD+Momentum** | 0.02 | beta=0.9 | **84.63% +/- 0.58%** | Fast |
| **Adam** | 0.001 | beta1=0.9, beta2=0.999 | **83.99% +/- 0.63%** | Fast |
| **AdamW** | 0.001 | decay=0.01 | **84.10% +/- 0.65%** | Fast |

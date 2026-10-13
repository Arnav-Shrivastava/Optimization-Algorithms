# Fashion-MNIST Benchmark Results (3 Seeds)

| Optimizer | Learning Rate | Other Config | Test Accuracy (Mean +/- Std) | Convergence Speed |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | 0.05 | None | **77.18% +/- 0.51%** | Slow |
| **SGD+Momentum** | 0.02 | beta=0.9 | **84.44% +/- 0.53%** | Fast |
| **Adam** | 0.001 | beta1=0.9, beta2=0.999 | **84.04% +/- 0.60%** | Fast |
| **AdamW** | 0.001 | decay=0.01 | **83.97% +/- 0.53%** | Fast |

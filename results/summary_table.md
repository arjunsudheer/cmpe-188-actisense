# PAMAP2 Model Comparison Table

| Model                         | Config            |   TRAIN Acc |   VAL Acc |   TEST Acc |   TEST F1 |   Latency (ms) |
|:------------------------------|:------------------|------------:|----------:|-----------:|----------:|---------------:|
| Logistic Regression (PyTorch) | normal            |      0.9642 |    0.889  |     0.9372 |    0.9374 |         0.1432 |
| CNN-GRU (Dua et al.)          | feature_selection |      0.9922 |    0.883  |     0.9364 |    0.9367 |         3.8866 |
| CNN-BiLSTM (Challa et al.)    | feature_selection |      0.996  |    0.8784 |     0.9353 |    0.9347 |         2.5854 |
| Random Forest                 | normal            |      0.9939 |  nan      |     0.9276 |    0.929  |        65.778  |
| CNN-BiLSTM (Challa et al.)    | normal            |      0.9987 |    0.887  |     0.9267 |    0.9282 |         2.3695 |
| Logistic Regression (PyTorch) | feature_selection |      0.9564 |    0.8914 |     0.9265 |    0.9263 |         0.213  |
| Random Forest                 | feature_selection |      0.993  |  nan      |     0.8977 |    0.9025 |        68.8888 |
| CNN-GRU (Dua et al.)          | normal            |      0.998  |    0.8877 |     0.8778 |    0.8815 |         4.3708 |
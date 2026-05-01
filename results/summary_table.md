# PAMAP2 Model Comparison Table

| Model                         | Config            |   TRAIN Acc |   VAL Acc |   TEST Acc |   TEST F1 |   Latency (ms) |
|:------------------------------|:------------------|------------:|----------:|-----------:|----------:|---------------:|
| CNN-GRU (Dua et al.)          | normal            |      0.9956 |    0.9051 |     0.9494 |    0.9494 |         2.3806 |
| CNN-BiLSTM (Challa et al.)    | normal            |      0.9683 |    0.8653 |     0.9381 |    0.9368 |         1.2974 |
| Logistic Regression (PyTorch) | normal            |      0.9635 |    0.8956 |     0.9328 |    0.9332 |         0.1229 |
| CNN-GRU (Dua et al.)          | feature_selection |      0.9668 |    0.8829 |     0.9314 |    0.9316 |         2.2311 |
| Random Forest                 | normal            |      0.9939 |  nan      |     0.9276 |    0.929  |        52.1832 |
| Logistic Regression (PyTorch) | feature_selection |      0.9676 |    0.8893 |     0.9267 |    0.9263 |         0.1303 |
| CNN-BiLSTM (Challa et al.)    | feature_selection |      0.9983 |    0.8786 |     0.9013 |    0.9034 |         1.771  |
| Random Forest                 | feature_selection |      0.993  |  nan      |     0.8977 |    0.9025 |        52.7086 |
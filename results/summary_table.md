# PAMAP2 Model Comparison Table

| Model                         | Config            |   TRAIN Acc |   VAL Acc |   TEST Acc |   TEST F1 |   Latency (ms) |
|:------------------------------|:------------------|------------:|----------:|-----------:|----------:|---------------:|
| CNN-GRU (Dua et al.)          | feature_selection |      0.9979 |    0.8774 |     0.9422 |    0.9424 |         2.6416 |
| Logistic Regression (PyTorch) | normal            |      0.9636 |    0.8946 |     0.9323 |    0.9321 |         0.1182 |
| Random Forest                 | normal            |      0.9939 |  nan      |     0.9276 |    0.929  |        54.2744 |
| Logistic Regression (PyTorch) | feature_selection |      0.9439 |    0.8862 |     0.9251 |    0.925  |         0.1434 |
| CNN-BiLSTM (Challa et al.)    | feature_selection |      0.9957 |    0.8773 |     0.9207 |    0.9224 |         1.6193 |
| CNN-GRU (Dua et al.)          | normal            |      0.9993 |    0.8891 |     0.9168 |    0.9203 |         2.139  |
| CNN-BiLSTM (Challa et al.)    | normal            |      0.9886 |    0.8574 |     0.8925 |    0.8944 |         1.8062 |
| Random Forest                 | feature_selection |      0.9947 |  nan      |     0.8842 |    0.8901 |        54.2582 |
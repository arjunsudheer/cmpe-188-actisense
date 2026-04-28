# PAMAP2 Model Comparison Table

| Model                         | Config            |   TRAIN Acc |   VAL Acc |   TEST Acc |   TEST F1 |   Latency (ms) |
|:------------------------------|:------------------|------------:|----------:|-----------:|----------:|---------------:|
| CNN-GRU (Dua et al.)          | feature_selection |      0.9976 |    0.8733 |     0.9541 |    0.9541 |         2.2103 |
| CNN-GRU (Dua et al.)          | normal            |      0.9976 |    0.8861 |     0.9428 |    0.9431 |         2.1099 |
| Logistic Regression (PyTorch) | normal            |      0.9733 |    0.8975 |     0.9403 |    0.9404 |         0.0891 |
| Random Forest                 | normal            |      0.9939 |  nan      |     0.9276 |    0.929  |        53.7609 |
| Logistic Regression (PyTorch) | feature_selection |      0.952  |    0.8882 |     0.9267 |    0.9268 |         0.162  |
| Random Forest                 | feature_selection |      0.9947 |  nan      |     0.8842 |    0.8901 |        57.6418 |
| CNN-BiLSTM (Challa et al.)    | feature_selection |      0.9967 |    0.8684 |     0.8366 |    0.8368 |         1.6495 |
| CNN-BiLSTM (Challa et al.)    | normal            |      0.9951 |    0.8802 |     0.8319 |    0.8305 |         1.4376 |
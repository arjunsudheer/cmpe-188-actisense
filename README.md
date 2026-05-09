# ActiSense

- **Project Title:** ActiSense
- **Team Members:** Arjun Sudheer
- **Contact Information:** [arjun.sudheer@sjsu.edu](mailto:arjun.sudheer@sjsu.edu)
- **GitHub Link:** [https://github.com/arjunsudheer/cmpe-188-actisense](https://github.com/arjunsudheer/cmpe-188-actisense)

This repository contains the final project for CMPE 188 at San Jose State University. **ActiSense** is a robust Human Activity Recognition (HAR) system that leverages wearable sensor data from the PAMAP2 dataset to classify physical activities with high precision and low latency.

## Problem Statement

Human Activity Recognition (HAR) is a critical task in ubiquitous computing, enabling applications in healthcare, sports monitoring, and elderly care. This project, **ActiSense**, leverages the PAMAP2 Physical Activity Monitoring dataset to build a robust system capable of classifying various physical activities using multi-modal sensor data.

In my final demo, I compare the performance of Logistic Regression, Random Forest, CNN-BiLSTM (Challa et al.), and CNN-GRU (Dua et al.) on the PAMAP2 dataset. This allows me to compare the performance of traditional machine learning with deep learning on time-series data. I compare the four models in terms of accuracy, precision, recall, and F1-score, and I also include a confusion matrix. I conducted exploratory data analysis by understanding the relationships between the features and the similarity between them using techniques like Principal Component Analysis (PCA). I also explored feature selection and compared model accuracy when performing feature selection versus training on all the data. I discussed the techniques I used for feature selection and how they affected the performance of all four models. For the real-time evaluation requirement, I simulated streaming sensor data by feeding samples one window at a time to measure prediction performance and inference latency. The goal of this project is to build an activity detection model that can classify real human physical activity from wearable sensor data, which enables health monitoring and fitness tracking capabilities.

## Dataset

I am using the **PAMAP2 Physical Activity Monitoring dataset** published by the UCI Machine Learning Repository.

[https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

### Dataset Information

The data was collected from **9 subjects** (8 male, 1 female) who performed 18 different physical activities. The data was captured at a sampling rate of **100 Hz** for the IMU sensors and approximately **9 Hz** for the heart rate monitor.

#### Features

The raw dataset contains 54 columns of information, which I have categorized and labeled as follows:

1. **Timestamp:** Sample index in seconds.
2. **Heart Rate:** Measured in beats per minute (bpm).
3. **IMU Sensors:** Three Inertial Measurement Units located on the **Hand**, **Chest**, and **Ankle** (51 features). Each IMU provides:
    - **Temperature:** Internal sensor temperature (*Dropped in preprocessing as it did not contribute to activity classification*).
    - **3D-Acceleration:** +/-16g range, providing high-magnitude movement data (m/s^2).
    - **3D-Acceleration:** +/-6g range, providing high-resolution movement data (m/s^2) (*Dropped in preprocessing due to lack of calibration*).
    - **3D-Gyroscope:** Angular velocity in rad/s.
    - **3D-Magnetometer:** Ambient magnetic field in \muT.
    - **Orientation (4):** Quaternion representation of orientation (*Dropped as they were invalid in this data collection*).

#### Labels

The dataset classifies activities into 12 core "Protocol" activities and several "Optional" activities:

- **Protocol:** Lying, Sitting, Standing, Walking, Running, Cycling, Nordic Walking, Ascending Stairs, Descending Stairs, Vacuum Cleaning, Ironing, Rope Jumping.
- **Optional:** Watching TV, Computer Work, Car Driving, Folding Laundry, House Cleaning, Playing Soccer.
- **Transient (0):** Non-activity periods (dropped during cleaning).

## Implementation

### 1. Dataset Preprocessing

The dataset preprocessing stage first extracts the data from the provided .dat files.

- **Cleaning:** Removal of the "transient" (0) label and invalid features (orientation, 6g accelerometer).
- **Missing Values / NAN:** Linear interpolation is used for the Heart Rate data to match the 100Hz IMU frequency, and forward/back-filling handles occasional wireless packet drops.
- **Normalization:** `StandardScaler` is applied to ensure all features contribute equally to the models.
- **Manual Feature Selection:** Testing experiments where high-noise or redundant features (specifically highly correlated magnetometers) are dropped based on EDA findings and Random Forest feature importance rankings.
- **Data Splitting:** Subjects are split into Train (101-105, 109), Validation (106, 107), and Test (108) to evaluate cross-subject generalization.

### 2. Sliding Window Creation (Time-series)

To capture movement patterns over time, the continuous data is divided into sliding windows:

- **Window Length:** 128 samples (~1.28 seconds of activity).
- **Overlap:** 64 samples (50%).

This transformation allows the LSTM and GRU models to learn temporal dependencies.

### 3. Exploratory Data Analysis (EDA)

I generate the following plots to understand the data before modeling:

- **Activity Distribution:** Visualizing class imbalance across different subjects.
- **Correlation Heatmaps:** Identifying redundant sensors and highly correlated feature groups.
- **Principal Component Analysis (PCA):** Visualizing activity clusters in 2D and assessing feature contributions to variance using Scree plots.
- **Sensor Snippets:** Plotting raw acceleration data for different intensities (e.g., Lying vs. Running) to verify signal quality.
- **Random Forest Feature Importance:** Ranking features to identify the most predictive signals (e.g., specific accelerometer axes) and validate manual feature selection.

### 4. Model Training and Evaluation

The following four models will be compared against each other. I compare Random Forest and Logistic Regression models against CNN-BiLSTM and CNN-GRU to compare traditional classifier models versus time-series based deep learning:

- **Random Forest:** A robust ensemble-based baseline.
- **Logistic Regression (PyTorch):** Optimized with weighted CrossEntropyLoss to handle imbalanced activity classes.
- **CNN-BiLSTM / CNN-GRU:** Advanced hybrid architectures that combine multi-scale Convolutional layers (kernels 3, 7, 11) for spatial feature extraction with Recurrent layers (BiLSTM or GRU) for temporal dependencies.

### Technical Highlights

1. **ML Feature Engineering:** For traditional models, I implemented a vectorized extraction pipeline that computes 18 features per channel (mean, std, min, max, RMS, skew, kurtosis, MAD, zero-crossing rate, and 8 frequency-domain features including spectral entropy and sub-band power) plus cross-axis correlations.
2. **Hybrid Deep Learning:** The DL models use a "TimeDistributed" approach, processing windows as sub-sequences to capture local patterns before aggregating temporal information.
3. **Real-Time Simulation:** Each model was benchmarked for inference latency by simulating a one-window-at-a-time data stream, ensuring feasibility for wearable deployment.
4. **Feature Importance Validation:** Used Random Forest importance scores to confirm that dropping redundant magnetometer axes maintained model performance while reducing feature dimensionality.

Performance is evaluated using Accuracy, Precision, Recall, and F1-Score, with confusion matrices and Precision-Recall curves generated for each experiment.

## Finished Features

### 1. **Data Preprocessing Pipeline**

- **Raw Data Extraction:** Parses PAMAP2 `.dat` files from the UCI repository and combines data from all 9 subjects.
- **Data Cleaning:**
  - Removes non-protocol activities (transient label 0) and invalid features (temperature, 6g accelerometer, quaternion orientation).
  - Keeps only the 12 protocol activities for the main analysis.
- **Missing Value Handling:**
  - Heart rate data (9 Hz) is interpolated to match the 100 Hz IMU sampling rate using linear interpolation.
  - Occasional wireless packet drops in sensor data are handled with forward-fill, backward-fill, and linear interpolation (max gap: 5 samples).
  - Residual NaN values are filled with column medians computed per subject.
- **Normalization:** StandardScaler applied to all features to ensure equal contribution to models.
- **Subject-Based Data Splitting:**
  - **Train:** Subjects 101, 102, 103, 104, 108, 109
  - **Validation:** Subjects 105, 106
  - **Test:** Subject 107 (cross-subject generalization evaluation)

### 2. **Sliding Window Creation**

Continuous time-series data is transformed into fixed-size windows to capture temporal patterns:

- **Window Size:** 128 samples (~1.28 seconds at 100 Hz sampling rate)
- **Overlap:** 64 samples (50% overlap) to maintain temporal continuity
- **Purity Threshold:** Windows are labeled with the majority activity, keeping only windows where the majority activity comprises ≥90% of the window.
- **Output:** Creates feature matrices for both machine learning (vectorized features) and deep learning (raw windowed sequences) pipelines.

### 3. **Exploratory Data Analysis (EDA)**

A comprehensive EDA pipeline generates five key visualizations to understand data characteristics before modeling:

- **Activity Distribution Plot:** Bar chart showing class distribution across the 12 protocol activities, revealing imbalance patterns that inform weighted loss function choices during training.
- **Sensor Correlation Heatmap:** 51×51 correlation matrix identifying redundant features (e.g., chest magnetic X/Z and ankle magnetic Y) that guided manual feature selection.
- **PCA Activity Clusters:** 2D scatter plot of activities in PCA space, visualizing separability between activities and identifying similar sensor signatures (e.g., sitting vs. standing).
- **PCA Variance / Scree Plot:** Explained variance ratio by principal component plus top 10 feature contributions to PC1, guiding dimensionality reduction decisions.
- **Sensor Snippets:** Time-series acceleration plots from 6 representative activities (lying, sitting, walking, running, ascending stairs, rope jumping) showing 3 lines per activity: **Hand X** (arm movement), **Chest X** (body forward/backward motion), and **Ankle X** (leg motion and stepping patterns).

### 4. **Feature Engineering for Machine Learning**

For traditional ML models (Random Forest, Logistic Regression), a vectorized feature extraction pipeline computes **18 features per sensor channel** from each 128-sample window:

- **Time-Domain Features (10 per channel):**
  - Mean, Standard Deviation, Min, Max
  - Range (Max - Min), Root Mean Square (RMS)
  - Skewness, Kurtosis, Mean Absolute Deviation (MAD), Zero-Crossing Rate
  
- **Frequency-Domain Features (8 per channel):**
  - Dominant Frequency and Dominant Power
  - Spectral Entropy (measures signal complexity)
  - Total Power (sum of all spectral components)
  - Sub-band Power in 4 frequency ranges: [0-1 Hz], [1-3 Hz], [3-10 Hz], [10-25 Hz]

- **Cross-Axis Correlations (9 additional features):**
  - Pearson correlations between X-Y, X-Z, and Y-Z acceleration axes for each of the three IMU locations (hand, chest, ankle).
  
**Total Features:**

- **Without Feature Selection:** 51 features × 18 features/channel + 9 cross-correlations = **927 features**
- **With Feature Selection (Top 10):** Reduced to **~100 features** by dropping redundant magnetometer axes

All features are standardized using StandardScaler to ensure equal contribution during model training.

### 5. **Model Training & Evaluation**

Four models are trained and compared:

- **Random Forest:** 100 trees, ensemble-based baseline for structured feature data.
- **Logistic Regression (PyTorch):** Optimized with weighted CrossEntropyLoss to handle class imbalance.
- **CNN-BiLSTM:** Hybrid architecture combining 1D CNNs (kernels: 3, 7, 11) with Bidirectional LSTM for temporal modeling.
- **CNN-GRU:** Similar hybrid design using Gated Recurrent Units instead of LSTM.

**Evaluation Metrics:**

- Accuracy, Precision, Recall, F1-Score
- Confusion matrices for per-activity performance
- Precision-Recall curves for each activity class

### 6. **Real-Time Inference Simulation**

Models are benchmarked for latency by simulating streaming sensor data:

- One window fed at a time to measure prediction speed.
- Ensures feasibility for real-time wearable deployment.

### 7. **Feature Importance Analysis**

Random Forest feature importance scores rank features by predictive power, validating manual feature selection choices and identifying which sensors/axes are most informative for activity classification.

### 8. **Cross-Subject Generalization Testing**

- Trained models are evaluated on a held-out subject (107) to assess generalization across different individuals.
- Results demonstrate the model's ability to recognize activities from new subjects not seen during training.

## Final Results

The models were evaluated on a held-out subject (107) to test cross-subject generalization. Results are shown for both the **Normal** dataset (all 51 features) and **Feature Selection** dataset (redundant magnetometer features removed).

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

### Key Findings

- **Best Overall Performance:** Logistic Regression achieves the highest accuracy (93.7%) on the full feature set with exceptional inference speed (0.14ms), making it ideal for resource-constrained wearable devices.
- **Feature Selection Trade-offs:** Removing redundant magnetometer features surprisingly improves CNN-GRU performance (88.8% → 93.6%) while slightly degrading Random Forest and CNN-BiLSTM (1-2% drop), indicating that feature reduction can benefit certain architectures more than others.
- **Deep Learning vs. Traditional ML:** Deep learning models (CNN-GRU, CNN-BiLSTM) exhibit higher sensitivity to feature selection, suggesting they benefit from the noise reduction despite losing some information, while Logistic Regression remains robust across both datasets. This is supported by the sensor correlation plot that shows the magnetometers and accelerometer having high correlation. More complex deep learning models that are more prone to overfitting may benefit from feature selection.
- **Real-Time Feasibility:** All models except Random Forest demonstrate sub-5ms latency, with Logistic Regression's 0.14ms making it suitable for real-time wearable applications even on battery-constrained devices.

## Limitations

The feature selection strategy (manually dropping correlated magnetometer axes) shows mixed results, with some models improving while others degrade. A more sophisticated feature selection technique (e.g., recursive feature elimination, genetic algorithms) may yield better generalization. Additionally, the dataset's relatively small size and potential subject-specific biases could limit cross-subject generalization beyond the tested hold-out subject.

## Conclusion

The project successfully demonstrates that robust activity recognition is achievable across multiple model architectures on the PAMAP2 dataset. While deep learning hybrids can achieve competitive performance, the Logistic Regression baseline offers a compelling trade-off between accuracy (93.7%) and real-time feasibility (0.14ms latency), making it the most practical choice for deployment on wearable devices where computational resources and battery life are critical constraints.

# ActiSense

This repository contains the final project for CMPE 188 at San Jose State University. **ActiSense** is a robust Human Activity Recognition (HAR) system that leverages wearable sensor data from the PAMAP2 dataset to classify physical activities with high precision and low latency.

## Team Members:

- Arjun Sudheer

## Problem Statement:

Human Activity Recognition (HAR) is a critical task in ubiquitous computing, enabling applications in healthcare, sports monitoring, and elderly care. This project, **ActiSense**, leverages the PAMAP2 Physical Activity Monitoring dataset to build a robust system capable of classifying various physical activities using multi-modal sensor data.

In my final demo, I compare the performance of Logistic Regression, Random Forest, CNN-BiLSTM (Challa et al.), and CNN-GRU (Dua et al.) on the PAMAP2 dataset. This allows me to compare the performance of traditional machine learning with deep learning on time-series data. I compare the four models in terms of accuracy, precision, recall, and F1-score, and I also include a confusion matrix. I conducted exploratory data analysis by understanding the relationships between the features and the similarity between them using techniques like Principal Component Analysis (PCA). I also explored feature selection and compared model accuracy when performing feature selection versus training on all the data. I discussed the techniques I used for feature selection and how they affected the performance of all four models. For the real-time evaluation requirement, I simulated streaming sensor data by feeding samples one window at a time to measure prediction performance and inference latency. The goal of this project is to build an activity detection model that can classify real human physical activity from wearable sensor data, which enables health monitoring and fitness tracking capabilities.

## Dataset

I am using the **PAMAP2 Physical Activity Monitoring dataset** published by the UCI Machine Learning Repository. 

[https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

### Dataset Information

The data was collected from **9 subjects** (8 male, 1 female) who performed 18 different physical activities. The data was captured at a sampling rate of **100 Hz** for the IMU sensors and approximately **9 Hz** for the heart rate monitor.

#### Features

The raw dataset contains 54 columns of information, which I have categorized and labeled as follows:

1.  **Timestamp:** Sample index in seconds.
2.  **Heart Rate:** Measured in beats per minute (bpm).
3.  **IMU Sensors:** Three Inertial Measurement Units located on the **Hand**, **Chest**, and **Ankle** (51 features). Each IMU provides:
    *   **Temperature:** Internal sensor temperature (*Dropped in preprocessing as it did not contribute to activity classification*).
    *   **3D-Acceleration:** +/-16g range, providing high-magnitude movement data (m/s^2).
    *   **3D-Acceleration:** +/-6g range, providing high-resolution movement data (m/s^2) (*Dropped in preprocessing due to lack of calibration*).
    *   **3D-Gyroscope:** Angular velocity in rad/s.
    *   **3D-Magnetometer:** Ambient magnetic field in \muT.
    *   **Orientation (4):** Quaternion representation of orientation (*Dropped as they were invalid in this data collection*).

#### Labels

The dataset classifies activities into 12 core "Protocol" activities and several "Optional" activities:
-   **Protocol:** Lying, Sitting, Standing, Walking, Running, Cycling, Nordic Walking, Ascending Stairs, Descending Stairs, Vacuum Cleaning, Ironing, Rope Jumping.
-   **Optional:** Watching TV, Computer Work, Car Driving, Folding Laundry, House Cleaning, Playing Soccer.
-   **Transient (0):** Non-activity periods (dropped during cleaning).

## Planned System Approach

### 1. Dataset Preprocessing

The dataset preprocessing stage first extracts the data from the provided .dat files.

-   **Cleaning:** Removal of the "transient" (0) label and invalid features (orientation, 6g accelerometer).
-   **Missing Values / NAN:** Linear interpolation is used for the Heart Rate data to match the 100Hz IMU frequency, and forward/back-filling handles occasional wireless packet drops.
-   **Normalization:** `StandardScaler` is applied to ensure all features contribute equally to the models.
-   **Manual Feature Selection:** Testing experiments where high-noise or redundant features (specifically highly correlated magnetometers) are dropped based on EDA findings and Random Forest feature importance rankings.
-   **Data Splitting:** Subjects are split into Train (101-105, 109), Validation (106, 107), and Test (108) to evaluate cross-subject generalization.

### 2. Sliding Window Creation (Time-series)

To capture movement patterns over time, the continuous data is divided into sliding windows:

-   **Window Length:** 128 samples (~1.28 seconds of activity).
-   **Overlap:** 64 samples (50%).

This transformation allows the LSTM and GRU models to learn temporal dependencies.

### 3. Exploratory Data Analysis (EDA)

I generate the following plots to understand the data before modeling:

-   **Activity Distribution:** Visualizing class imbalance across different subjects.
-   **Correlation Heatmaps:** Identifying redundant sensors and highly correlated feature groups.
-   **Principal Component Analysis (PCA):** Visualizing activity clusters in 2D and assessing feature contributions to variance using Scree plots.
-   **Sensor Snippets:** Plotting raw acceleration data for different intensities (e.g., Lying vs. Running) to verify signal quality.
-   **Random Forest Feature Importance:** Ranking features to identify the most predictive signals (e.g., specific accelerometer axes) and validate manual feature selection.

### 4. Model Training and Evaluation

The following four models will be compared against each other. I compare Random Forest and Logistic Regression models against CNN-BiLSTM and CNN-GRU to compare traditional classifier models versus time-series based deep learning:

-   **Random Forest:** A robust ensemble-based baseline.
-   **Logistic Regression (PyTorch):** Optimized with weighted CrossEntropyLoss to handle imbalanced activity classes.
-   **CNN-BiLSTM / CNN-GRU:** Advanced hybrid architectures that combine multi-scale Convolutional layers (kernels 3, 7, 11) for spatial feature extraction with Recurrent layers (BiLSTM or GRU) for temporal dependencies.

### Technical Highlights

1.  **ML Feature Engineering:** For traditional models, I implemented a vectorized extraction pipeline that computes 18 features per channel (mean, std, min, max, RMS, skew, kurtosis, MAD, zero-crossing rate, and 8 frequency-domain features including spectral entropy and sub-band power) plus cross-axis correlations.
2.  **Hybrid Deep Learning:** The DL models use a "TimeDistributed" approach, processing windows as sub-sequences to capture local patterns before aggregating temporal information.
3.  **Real-Time Simulation:** Each model was benchmarked for inference latency by simulating a one-window-at-a-time data stream, ensuring feasibility for wearable deployment.
4.  **Feature Importance Validation:** Used Random Forest importance scores to confirm that dropping redundant magnetometer axes maintained model performance while reducing feature dimensionality.

Performance is evaluated using Accuracy, Precision, Recall, and F1-Score, with confusion matrices and Precision-Recall curves generated for each experiment.

## Current Implementation Progress

The project is complete.

### Final Results & Comparison

The models were evaluated on a held-out subject (108) to test cross-subject generalization. The results show that while the CNN-GRU hybrid model provides the highest classification accuracy, the Logistic Regression model offers an exceptional balance of performance and efficiency.

| Model | Dataset Config | Test F1 | Latency (ms) |
| :--- | :--- | :---: | :---: |
| **CNN-GRU (Dua et al.)** | Normal | 0.9494 | 2.38 |
| **CNN-BiLSTM (Challa et al.)** | Normal | 0.9368 | 1.30 |
| **Logistic Regression** | Normal | 0.9332 | 0.12 |
| **Random Forest** | Normal | 0.9290 | 52.18 |

### Conclusion

The project successfully demonstrates that deep learning hybrids like CNN-GRU are highly effective for HAR tasks on the PAMAP2 dataset, achieving over 94% F1-score. Furthermore, the extremely low latency of the Logistic Regression model (0.12ms) highlights its suitability for resource-constrained wearable devices where real-time performance is paramount. All project objectives, including EDA, feature selection, model comparison, and real-time simulation, have been fully met.

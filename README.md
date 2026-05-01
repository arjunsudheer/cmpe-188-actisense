# ActiSense

This repository is dedicated to the ActiSense final project for CMPE 188 at San Jose State University. This project focuses on Human Activity Recognition (HAR) using wearable sensor data.

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
    *   **Temperature:** Internal sensor temperature in celsius.
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
-   **Manual Feature Selection:** Testing experiments where high-noise or low-importance features (like gyroscopes) are dropped based on EDA.
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

### 4. Model Training and Evaluation

The following four models will be compared against each other. I compare Random Forest and Logistic Regression models against CNN-BiLSTM and CNN-GRU to compare traditional classifier models versus time-series based deep learning:

-   **Random Forest:** A robust ensemble-based baseline.
-   **Logistic Regression (PyTorch):** Optimized with weighted CrossEntropyLoss to handle imbalanced activity classes.
-   **CNN-BiLSTM / CNN-GRU:** Advanced hybrid architectures that combine Convolutional layers for spatial feature extraction with Recurrent layers for temporal dependencies.

Performance is evaluated using Accuracy, Precision, Recall, and F1-Score, with confusion matrices and Precision-Recall curves generated for each experiment.

## Current Implementation Progress

Currently, the project is mostly complete. I have implemented the full preprocessing pipeline, conducted thorough EDA, and trained all four models. The system supports both the full dataset and a manual feature selection configuration (excluding gyroscope data). I have also implemented a real-time simulation to measure inference latency for each model. My next step is to extract the Random Forest Classifier feature importance score to justify the manual feature selection.

### Final Results

The models were evaluated on a held-out subject (108). The results demonstrate that the CNN-GRU hybrid model achieves the highest F1-score, while Logistic Regression provides the lowest inference latency.

| Model | Dataset Config | Test F1 | Latency (ms) |
| :--- | :--- | :---: | :---: |
| **CNN-GRU (Dua et al.)** | Feature Selection | 0.9424 | 2.64 |
| **Logistic Regression** | Normal | 0.9321 | 0.12 |
| **Random Forest** | Normal | 0.9290 | 54.27 |
| **CNN-BiLSTM (Challa et al.)** | Feature Selection | 0.9224 | 1.62 |

The project successfully achieved its goal of classifying physical activities with high accuracy while maintaining low enough latency for real-time applications.

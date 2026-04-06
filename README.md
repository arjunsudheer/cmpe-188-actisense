# ActiSense

This repository is dedicated to the ActiSense final project for CMPE 188 at San Jose State University. This project focuses on Human Activity Recognition (HAR) using wearable sensor data.

## Team Members:

- Arjun Sudheer

## Problem Statement:

Human Activity Recognition (HAR) is a critical task in ubiquitous computing, enabling applications in healthcare, sports monitoring, and elderly care. This project, **ActiSense**, leverages the PAMAP2 Physical Activity Monitoring dataset to build a robust system capable of classifying various physical activities using multi-modal sensor data.

In my final demo, I will compare the performance of Logistic Regression, Random Forest, Long-Short Term Memory (LSTM), and Gradient Recurrent Unit (GRU) on the PAMAP2 dataset. This allows me to compare the performance of traditional machine learning with deep learning on time-series data. I will compare the four models in terms of accuracy, precision, recall, and F1-score, and I will also include a confusion matrix. I will conduct exploratory data analysis by understanding the relationships between the features and the similarity between them using techniques like Principal Component Analysis (PCA). I will also explore feature selection and compare model accuracy when performing feature selection versus training on all the data. I will discuss the techniques I used for feature selection and how they affected the performance of all four models. For the real-time evaluation requirement, I will simulate streaming sensor data by feeding samples one window at a time to measure prediction performance and inference latency. The goal of this project is to build an activity detection model that can classify real human physical activity from wearable sensor data, which enables health monitoring and fitness tracking capabilities.

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

The following four models will be compared against each other. I compare Random Forest and Logistic Regression models against LSTM and GRU to compare traditional classifier models veruses time-series based classification:

-   **Random Forest:** A robust ensemble-based baseline.
-   **Logistic Regression (PyTorch):** Optimized with weighted CrossEntropyLoss to handle imbalanced activity classes.
-   **LSTM / GRU:** Deep learning models tailored for sequential time-series data.

Performance is evaluated using Accuracy, Precision, Recall, and F1-Score, with confusion matrices and Precision-Recall curves generated for each experiment.

## Current Implementation Progress

Currently, I have the dataset preprocessing, EDA plots, Random Forest Classifier, and Logistic Regression model implemented. I have made these work with both the normal and manual feature selection datasets. I have also selected some features to manually drop based on the EDA plots for further analysis.

### Preliminary Results

The current models have been evaluated on the held-out subject (108). Initial F1-scores are summarized below:

| Model | Normal Dataset | Feature Selection (No Gyro) |
| :--- | :---: | :---: |
| **Random Forest** | 0.4148 | 0.3910 |
| **Logistic Regression** | 0.3069 | 0.3114 |

My next steps include implementing the LSTM and GRU models, and implementing data streaming for live analysis.

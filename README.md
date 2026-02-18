# 🚗 Driver Behavior Profiling: Unsupervised Telematics Analysis

> **Status:** Completed ✅ | **Validation Score:** 1.00 ARI 🏆

## 📖 Overview
This project builds a Machine Learning pipeline to automatically categorize drivers into distinct behavioral profiles (**Safe**, **Aggressive**, **Distracted**) using raw telematics data.

**The "Real-World" Twist:**
Although the original dataset contained labeled data (ground truth), **this project intentionally treats the problem as Unsupervised Learning.** In a real-world telematics scenario, we rarely have pre-labeled data telling us "User X is aggressive." We only have raw sensor logs. This project demonstrates how to use **Dimensionality Reduction (PCA)** and **Clustering (K-Means)** to mathematically discover these behaviors from scratch, using the hidden labels only for final validation.

## 🎯 Objectives
1.  **Simulate Real-World Conditions:** Train a model without access to the target variable (`driver_label`).
2.  **Dimensionality Reduction:** Use Principal Component Analysis (PCA) to compress noisy sensor data into meaningful "Behavioral Dimensions."
3.  **Automatic Profiling:** Use K-Means clustering to group drivers based on their PCA coordinates.
4.  **Validation:** Compare the unsupervised clusters against the hidden ground truth to measure the model's ability to "recover" the true labels.

## 🛠️ The Workflow

### 1. Data Preprocessing & Correlation Analysis
We analyzed the correlation between 11 telematics sensors, including:
* `speed_kmph`, `accel_x` (Longitudinal), `accel_y` (Lateral)
* `brake_pressure`, `throttle`
* `phone_usage`, `reaction_time`
* `steering_angle`

**Key Insight:** We identified an **"Aggression Box"** where Speed, Acceleration, and Braking were highly multicollinear, suggesting they could be compressed into a single feature.

### 2. Dimensionality Reduction (PCA)
We used **Principal Component Analysis (PCA)** to reduce the dataset to **3 components**, explaining **~80%** of the variance.

| Component | Interpretation | Key Features |
| :--- | :--- | :--- |
| **PC1** | **Aggression** 😡 | High `throttle`, `brake_pressure`, `accel_x` |
| **PC2** | **Distraction** 📱 | High `phone_usage`, `reaction_time`, `lane_deviation` |
| **PC3** | **Geometry** 📐 | Purely `steering_angle` (Independent of behavior) |

### 3. Unsupervised Clustering (K-Means)
We applied **K-Means Clustering (k=3)** on the PCA-transformed data. The algorithm automatically segmented drivers into three distinct groups:
1.  **Cluster 0:** Safe/Chill Drivers (Low Aggression, Low Distraction)
2.  **Cluster 1:** Distracted Drivers (High Phone Usage)
3.  **Cluster 2:** Aggressive Drivers (High Speed/Braking)

## 🏆 Results & Validation

Since we withheld the labels during training, we used the **Adjusted Rand Index (ARI)** to compare our unsupervised clusters against the actual hidden labels.

`
Adjusted Rand Score: 1.00
Homogeneity Score:   1.00
`

**Conclusion**: The model achieved a perfect 1.0 score, proving that:

- The driver behaviors are mathematically distinct and linearly separable in the PCA space.

- We can effectively auto-label new drivers with 100% accuracy without needing a human to supervise the training process.

## 📊 Visualizations
### 3D Cluster Analysis
The 3D scatter plot reveals three perfectly separated islands of driver behavior.

### The "Aggression Box" Correlation
Heatmap showing the intense correlation between braking, throttling, and G-forces.



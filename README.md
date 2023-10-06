# Parkinson_Disease_Diagnosis

EEG Signal Analysis for Parkinson's Disease Diagnosis
Overview
This repository contains the code and documentation for a bachelor's thesis project focused on analyzing EEG signals to diagnose Parkinson's disease. The project aims to classify EEG data into two groups: patients with Parkinson's disease and a control group.

Dataset
The dataset used in this project is publicly available and was recorded at the University of Iowa. It includes EEG data from two classes: the patient group and the control group. Information about whether the patients were under medication is not provided, so the patient group includes individuals both with and without medication. The EEG data was recorded using AgCl/Ag porous electrodes at a sampling frequency of 500 Hz, with 63 channels, and each recording lasts approximately 2 minutes.

Preprocessing
Filtering
The preprocessing steps include filtering the EEG data. A bandpass filter from 0.5 Hz to 50 Hz is applied, and the sampling rate is reduced from 500 Hz to 100 Hz.

Segmentation
The data is segmented into smaller windows. Each window contains 1000 data points to capture disease-related information while maintaining a reasonable number of samples for classification. Initial trials from each subject are excluded to improve classifier performance.

Normalization
For each channel, the feature extraction process is followed by normalization with respect to the mean and standard deviation of the data. This ensures that each feature has zero mean and unit variance.

Feature Selection
The project employs various feature selection techniques to identify the most informative features for classification:

Square-Chi: This method measures the independence between two categorical variables and selects features accordingly.

Score Fisher: It ranks features based on Fisher's discriminant score, indicating the separability of classes.

Coefficient Correlation: Features are selected based on the correlation matrix of the data.

PCA (Principal Component Analysis): PCA is used to reduce the dimensionality of the data while preserving as much variance as possible.

NCA (Neighborhood Component Analysis): NCA focuses on selecting features that maximize the separation between classes based on nearest neighbors.

Classification
The project employs several machine learning classifiers for EEG signal classification, including Support Vector Machines (SVM), Multi-Layer Perceptrons (MLP), and Random Forest. The classification results are reported in terms of accuracy and AUC (Area Under the ROC Curve).

Repository Structure
data/: Placeholder for the dataset or data preprocessing scripts.
code/: Contains the code for data preprocessing, feature extraction, feature selection, and classification.
results/: Stores the classification results, including accuracy and AUC scores.
docs/: Documentation and research papers related to the project.
Usage
To reproduce the results or run the analysis on a different dataset, follow the code provided in the code/ directory. Be sure to configure the paths to the dataset and adjust hyperparameters as needed.

Conclusion
This project explores EEG signal analysis for the diagnosis of Parkinson's disease. It showcases different preprocessing techniques, feature selection methods, and machine learning classifiers to achieve accurate classification results. The research findings can be valuable for further studies in the field of medical diagnostics using EEG signals.

You can modify and expand this README to include specific details about your project, such as installation instructions, dependencies, and additional documentation links.

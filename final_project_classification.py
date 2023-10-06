# -*- coding: utf-8 -*-
"""Final_project_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gPNca-F_FWCYN9AvfLXW_0clnRZtYf9D

####***Alireza Alinejad 96102031***
***Mentor: Dr. Shamsolahi***

*Proccesing and Classification of EEG - Bachelor Thesis*

###**Imports**
"""

!pip install ncafs
!pip install skfeature-chappers
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from scipy.signal import butter, lfilter , freqz
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFE, f_classif
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.svm import SVR
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from ncafs import NCAFSC

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/EEG_Project/Data_main

!ls

"""###**Loading Data**"""

features = loadmat('all_features.mat')
features.keys()

data = features['features']
data_amplitude = np.abs(data)
# Replace the complex numbers in data with their amplitudes
data = data_amplitude
data.shape

# Define the number of samples and patients
num_samples = data.shape[0]
num_patients = 14
samples_per_patient = 12

# Define the number of samples for testing
num_test_samples = 24

# Calculate the number of samples for training
num_train_samples = num_samples - num_test_samples

# Define the labels for patients (1) and healthy people (0)
patient_labels = np.ones((num_patients * samples_per_patient,))
healthy_labels = np.zeros((num_patients * samples_per_patient,))

# Concatenate the labels for all samples
all_labels = np.concatenate((patient_labels, healthy_labels))

# Generate indices for shuffling the data
indices = np.random.permutation(num_samples)

# Shuffle the data and labels
shuffled_data = data[indices]
shuffled_labels = all_labels[indices]

# Split the data and labels into training and testing sets
train_data = shuffled_data[:num_train_samples]
train_labels = shuffled_labels[:num_train_samples]
test_data = shuffled_data[-num_test_samples:]
test_labels = shuffled_labels[-num_test_samples:]

# Display the shapes of train_data, train_labels, test_data, and test_labels
train_data_shape = train_data.shape
train_labels_shape = train_labels.shape
test_data_shape = test_data.shape
test_labels_shape = test_labels.shape

print("Shape of train_data:", train_data_shape)
print("Shape of train_labels:", train_labels_shape)
print("Shape of test_data:", test_data_shape)
print("Shape of test_labels:", test_labels_shape)

# Initialize empty arrays for class 1 and class 0 data
data_class_1 = []
data_class_0 = []

# Iterate through the training data and labels
for sample, label in zip(train_data, train_labels):
    if label == 1:
        data_class_1.append(sample)
    else:
        data_class_0.append(sample)

# Convert the lists to NumPy arrays
data_class_1 = np.array(data_class_1)
data_class_0 = np.array(data_class_0)

# Display the shapes of data_class_1 and data_class_0
data_class_1_shape = data_class_1.shape
data_class_0_shape = data_class_0.shape

print("Shape of data_class_1:", data_class_1_shape)
print("Shape of data_class_0:", data_class_0_shape)

# Create arrays for class 1 and class 0 labels
labels_class_1 = np.ones((data_class_1.shape[0],))
labels_class_0 = np.zeros((data_class_0.shape[0],))
y_label = np.concatenate((labels_class_1,labels_class_0), axis=0)
train_data = np.concatenate((data_class_0,data_class_1), axis=0)

# Display the shapes of the arrays
print("Shape of data:", data.shape)
print("Shape of data_class_1:", data_class_1.shape)
print("Shape of data_class_0:", data_class_0.shape)
print("Shape of train_data:", train_data.shape)
print("Shape of train_labels:", train_labels.shape)
print("Shape of test_data:", test_data.shape)

"""###**Standardize the Train and Test data**###"""

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.fit_transform(test_data)
# Display the maximum values, shapes of train_scaled and test_scaled
train_max = np.max(train_scaled)
test_max = np.max(test_scaled)
train_shape = train_scaled.shape
test_shape = test_scaled.shape

print("Maximum value in train_scaled:", train_max)
print("Maximum value in test_scaled:", test_max)
print("Shape of train_scaled:", train_shape)
print("Shape of test_scaled:", test_shape)

"""###**Training The Model**"""

x_train = train_data
y_train = y_label

# Set the number of folds for cross-validation
k_folds = 5

n_selected = 2;
# Define the feature extraction methods
methods = {
     'Chi Square': SelectKBest(chi2, k= 3),
      'Fisher Score': SelectKBest(f_classif, k= 2),
      'Correlation Coefficient': SelectKBest(lambda X, y: np.array(list(map(lambda x: np.corrcoef(x, y)[0, 1], X.T))).T, k= 3),
     'PCA': PCA(n_components= 3),
    'NCA': NeighborhoodComponentsAnalysis(n_components= 2),
}

# Define the classifiers
classifiers = {
    'SVM': SVC(kernel='linear', C=1, gamma='auto',probability=True),
    'MLP': MLPClassifier(hidden_layer_sizes=(200,), activation='relu', max_iter=200),
    'Random Forest': RandomForestClassifier(),
}

# Apply non-negative transformation to x_train
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_transformed = scaler.fit_transform(x_train)

# Initialize result dictionaries
confusion_matrices = {}
auc_scores = {}
accuracies = {}
fprs = {}
tprs = {}

# Perform feature extraction and classification for each method and classifier
for method_name, method in methods.items():
    x_train_feature_extracted = method.fit_transform(x_train_transformed, y_train)
    confusion_matrices[method_name] = {}
    auc_scores[method_name] = {}
    accuracies[method_name] = {}
    fprs[method_name] = {}
    tprs[method_name] = {}

    for classifier_name, classifier in classifiers.items():
        # Perform k-fold cross-validation
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
        y_pred_list = []
        y_true_list = []
        probas_list = []

        for train_index, val_index in skf.split(x_train_feature_extracted, y_train):
            x_train_fold, x_val_fold = x_train_feature_extracted[train_index], x_train_feature_extracted[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            classifier.fit(x_train_fold, y_train_fold)
            y_pred_fold = classifier.predict(x_val_fold)
            probas_fold = classifier.predict_proba(x_val_fold)[:, 1]

            y_pred_list.extend(y_pred_fold)
            y_true_list.extend(y_val_fold)
            probas_list.extend(probas_fold)

        # Calculate evaluation metrics
        confusion_matrices[method_name][classifier_name] = confusion_matrix(y_true_list, y_pred_list)
        auc_scores[method_name][classifier_name] = roc_auc_score(y_true_list, probas_list)
        accuracies[method_name][classifier_name] = accuracy_score(y_true_list, y_pred_list)
        fpr, tpr, _ = roc_curve(y_true_list, probas_list)
        fprs[method_name][classifier_name] = fpr
        tprs[method_name][classifier_name] = tpr
# # Visualize confusion matrices as heatmaps
# for method_name, method in methods.items():
#     for classifier_name, classifier in classifiers.items():
#         cm = confusion_matrices[method_name][classifier_name]
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, cmap='Reds', fmt='g', cbar=False)
#         plt.title(f'Confusion Matrix: {method_name} - {classifier_name}')
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.show()

# Display AUC scores
for method_name, method in methods.items():
    for classifier_name, classifier in classifiers.items():
        auc = auc_scores[method_name][classifier_name]
        print(f'AUC Score: {method_name} - {classifier_name}: {auc}')

# Display accuracies
for method_name, method in methods.items():
    for classifier_name, classifier in classifiers.items():
        acc = accuracies[method_name][classifier_name]
        print(f'Accuracy: {method_name} - {classifier_name}: {acc}')

# Plot ROC curves
for method_name, method in methods.items():
    for classifier_name, classifier in classifiers.items():
        fpr = fprs[method_name][classifier_name]
        tpr = tprs[method_name][classifier_name]
        auc = auc_scores[method_name][classifier_name]
        plt.plot(fpr, tpr, label=f'{method_name} - {classifier_name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

"""###**Test data**"""

# Apply feature extraction and classification on test data
x_test_transformed = scaler.transform(test_data)
x_test_feature_extracted = {}

for method_name, method in methods.items():
    x_test_feature_extracted[method_name] = method.transform(x_test_transformed)

for method_name, method in methods.items():
    for classifier_name, classifier in classifiers.items():
        # Train the classifier on the transformed training data
        x_train_feature_extracted = method.transform(x_train_transformed)
        classifier.fit(x_train_feature_extracted, y_train)

        # Make predictions on the test data
        x_test_method = x_test_feature_extracted[method_name]
        y_pred = classifier.predict(x_test_method)
        probas = classifier.predict_proba(x_test_method)[:, 1]

        # Calculate evaluation metrics
        confusion_matrix_test = confusion_matrix(test_labels, y_pred)
        auc_score_test = roc_auc_score(test_labels, probas)
        accuracy_test = accuracy_score(test_labels, y_pred)

        # Print evaluation metrics
        print(f'Test Results - {method_name} - {classifier_name}:')
        print(f'Confusion Matrix:\n{confusion_matrix_test}')
        print(f'AUC Score: {auc_score_test}')
        print(f'Accuracy: {accuracy_test}\n')

"""##AUC Curves for train"""

import io
import sys
import matplotlib.pyplot as plt

# Create empty variables or files to store the output
confusion_matrices_output = {}
auc_scores_output = {}
accuracies_output = ""

# Redirect print statements to capture the output
stdout = sys.stdout
sys.stdout = io.StringIO()

# ...

# Plot ROC curves
roc_curves = []
for method_name, method in methods.items():
    for classifier_name, classifier in classifiers.items():
        fpr = fprs[method_name][classifier_name]
        tpr = tprs[method_name][classifier_name]
        auc = auc_scores[method_name][classifier_name]

        plt.plot(fpr, tpr, label=f'{method_name} - {classifier_name} (AUC = {auc:.2f})')
        roc_curves.append(f'roc_{method_name}_{classifier_name}.png')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Save the AUC curves as image files
for curve, Results in zip(roc_curves, roc_curves):
    plt.savefig(Results)
    plt.close()

# ...

# Restore stdout
sys.stdout = stdout

# Print the captured output
print("AUC Scores:")
for method_classifier, auc in auc_scores_output.items():
    print(f'{method_classifier}: {auc}')

print("\nAccuracies:")
print(accuracies_output)

"""###Saving Heatmap of test"""

# Apply feature extraction and classification on test data
x_test_transformed = scaler.transform(test_data)
x_test_feature_extracted = {}

for method_name, method in methods.items():
    x_test_feature_extracted[method_name] = method.transform(x_test_transformed)

for method_name, method in methods.items():
    for classifier_name, classifier in classifiers.items():
        # Train the classifier on the transformed training data
        x_train_feature_extracted = method.transform(x_train_transformed)
        classifier.fit(x_train_feature_extracted, y_train)

        # Make predictions on the test data
        x_test_method = x_test_feature_extracted[method_name]
        y_pred = classifier.predict(x_test_method)
        probas = classifier.predict_proba(x_test_method)[:, 1]

        # Calculate evaluation metrics
        confusion_matrix_test = confusion_matrix(test_labels, y_pred)
        auc_score_test = roc_auc_score(test_labels, probas)
        accuracy_test = accuracy_score(test_labels, y_pred)

        # Print evaluation metrics
        print(f'Test Results - {method_name} - {classifier_name}:')
        print(f'Confusion Matrix:\n{confusion_matrix_test}')
        print(f'AUC Score: {auc_score_test}')
        print(f'Accuracy: {accuracy_test}\n')

        # Visualize confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_test, annot=True, cmap='Reds', fmt='g', cbar=False)
        plt.title(f'Confusion Matrix: {method_name} - {classifier_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'heatmap_test_{method_name}_{classifier_name}.png')  # Save the heatmap as an image file
        plt.close()  # Close the figure to release memory

"""###**saving results**

###**for stop overfitting**
"""

# Define the SVM classifier with regularization
svm_classifier = SVC(kernel='rbf', C=3.0, probability=True)

# Apply feature extraction and classification for SVM with regularization
for method_name, method in methods.items():
    x_train_feature_extracted = method.fit_transform(x_train_transformed, y_train)
    x_test_feature_extracted = method.transform(x_test_transformed)

    # Train the SVM classifier with regularization
    svm_classifier.fit(x_train_feature_extracted, y_train)

    # Make predictions on the test data
    y_pred_svm = svm_classifier.predict(x_test_feature_extracted)
    probas_svm = svm_classifier.predict_proba(x_test_feature_extracted)[:, 1]

    # Calculate evaluation metrics for SVM
    confusion_matrix_svm = confusion_matrix(test_labels, y_pred_svm)
    auc_score_svm = roc_auc_score(test_labels, probas_svm)
    accuracy_svm = accuracy_score(test_labels, y_pred_svm)

    # Print evaluation metrics for SVM
    print(f'Test Results - {method_name} - SVM:')
    print(f'Confusion Matrix:\n{confusion_matrix_svm}')
    print(f'AUC Score: {auc_score_svm}')
    print(f'Accuracy: {accuracy_svm}\n')

from sklearn.neural_network import MLPClassifier

# Define the MLP classifier with regularization
mlp_classifier = MLPClassifier(hidden_layer_sizes=(400,),activation='relu', alpha=0.001, max_iter=300)

# Apply feature extraction and classification for MLP with regularization
for method_name, method in methods.items():
    x_train_feature_extracted = method.fit_transform(x_train_transformed, y_train)
    x_test_feature_extracted = method.transform(x_test_transformed)

    # Train the MLP classifier with regularization
    mlp_classifier.fit(x_train_feature_extracted, y_train)

    # Make predictions on the test data
    y_pred_mlp = mlp_classifier.predict(x_test_feature_extracted)
    probas_mlp = mlp_classifier.predict_proba(x_test_feature_extracted)[:, 1]

    # Calculate evaluation metrics for MLP
    confusion_matrix_mlp = confusion_matrix(test_labels, y_pred_mlp)
    auc_score_mlp = roc_auc_score(test_labels, probas_mlp)
    accuracy_mlp = accuracy_score(test_labels, y_pred_mlp)

    # Print evaluation metrics for MLP
    print(f'Test Results - {method_name} - MLP:')
    print(f'Confusion Matrix:\n{confusion_matrix_mlp}')
    print(f'AUC Score: {auc_score_mlp}')
    print(f'Accuracy: {accuracy_mlp}\n')

"""#**Feature Selection Methods**

###**1. Apply Chi-square Test for dimensionality reduction**
"""

# x_train_chi2 = x_train + np.abs(np.min(x_train)) + 1
# x_train_cat = x_train_chi2.astype(int)
# chi2_features = SelectKBest(chi2, k=10)
# x_train_chi2_features = chi2_features.fit_transform(x_train_cat, y_train)
# # Display the shapes of x_train_chi2_features and x_train_cat
# x_train_chi2_features_shape = x_train_chi2_features.shape
# x_train_cat_shape = x_train_cat.shape

# print("Shape of x_train_chi2_features:", x_train_chi2_features_shape)
# print("Shape of x_train_cat:", x_train_cat_shape)

"""###**2. Apply Fisher’s Score for dimensionality reduction**

###**Fisher score**
"""

def f_score(X, y, f):
    """
    This function implements the anova f_value feature selection (existing method for classification in scikit-learn),
    where f_score = sum((ni/(c-1))*(mean_i - mean)^2)/((1/(n - c))*sum((ni-1)*std_i^2))
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y : {numpy array},shape (n_samples,)
        input class labels
    Output
    ------
    F: {numpy array}, shape (n_features,)
        f-score for each feature
    """
    F, pval = f_classif(X, y)
    """
    Rank features in descending order according to f-score, the higher the f-score, the more important the feature is
    """
    idx = np.argsort(F)
    ranks_fisher = idx[::-1]
    x_train_fisher_features = x_train[:,ranks_fisher[0:f]]
    return x_train_fisher_features

n_fisher = 10
x_train_fisher_features = f_score(x_train, y_train, n_fisher)
x_train_fisher_features.shape

"""###**3. Apply Correlation Coefficient for dimensionality reduction**"""

# from scipy.stats import pearsonr
# # Calculate the correlation coefficients
# correlation_coeffs = []
# for feature in range(x_train.shape[1]):
#     corr, _ = pearsonr(x_train[:, feature], y_train)
#     correlation_coeffs.append(abs(corr))

# # Sort the correlation coefficients and select the top-k features
# k = 10  # Number of features to select
# selected_feature_indices = np.argsort(correlation_coeffs)[-k:]

# # Create a new feature matrix with only the selected features
# x_train_reduced = x_train[:, selected_feature_indices]

# print("Transformed data shape:", x_train_reduced.shape)

"""###**4. Apply PCA for dimensionality reduction**"""

# n_pca = 10;
# pca = PCA(n_components=n_pca)
# x_train_pca_features = pca.fit_transform(x_train)
# print("Transformed data shape:", x_train_pca_features.shape)

"""###**5. Apply NCA for dimensionality reduction**"""

# n_nca = 10
# nca = NeighborhoodComponentsAnalysis(n_components=n_nca)
# eeg_data_nca_fit = nca.fit(x_train, y_train)
# eeg_data_nca = nca.transform(x_train)
# print("Original data shape:", x_train.shape)
# print("Transformed data shape:", eeg_data_nca.shape)

"""###**6. Apply recursive feature elimination (RFE) for dimensionality reduction**"""

# #estimator = RandomForestClassifier()
# estimator = SVC(kernel='linear', C=5, gamma='auto')
# n_REF = 4
# selector = RFE(estimator, n_features_to_select=n_REF, step=1)
# selector.fit(x_train, y_train.ravel())
# x_train_selected = selector.transform(x_train)
# x_val_selected = selector.transform(x_val)

# estimator.fit(x_train_selected, y_train)
# y_pred = estimator.predict(x_val_selected)
# y_pred
# accuracy = accuracy_score(y_val, y_pred)
# confusion_mat = confusion_matrix(y_val, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Confusion matrix:\n{confusion_mat}")

"""###**Define the SVM model**###"""

# svm = SVC(kernel='rbf', C=5, gamma='auto')

"""###**Define the pipeline to combine preprocessing and classification steps (SVM)**###"""

# features = [('pca', PCA(n_components=n_pca)),('nca', NeighborhoodComponentsAnalysis(n_components =n_nca))
#             ]
# # for i in range(1):
# pipeline = Pipeline([('scaler', StandardScaler()),
#                      (features[1]),
#                     ('svm', svm)])
# print(pipeline)

# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(svm, x_train_fisher_features, y_train.ravel(), cv=kfold, verbose=True, n_jobs=-1,
#                          error_score='raise', scoring='accuracy')
# print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean() * 100, scores.std() * 100))

"""###**Train and test the SVM using k-fold cross-validation**###"""

# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(pipeline, x_train, y_train.ravel(), cv=kfold, verbose=True, n_jobs=-1,
#                          error_score='raise', scoring='accuracy')
# print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean() * 100, scores.std() * 100))

# svm = SVC(kernel='rbf', C=5, gamma='auto')

# pipeline = Pipeline([('scaler', StandardScaler()),
#                      #('pca', PCA(n_components=3)),
#                      ('nca', NeighborhoodComponentsAnalysis(n_components =n_nca)),
#                      ('svm', svm)])
# pipeline.fit(x_train,y_train)
# # pipeline.fit(np.ones(x_train.shape), y_train)
# y_test_pred = pipeline.predict(x_val)
# accuracy = accuracy_score(y_val, y_test_pred)
# cm = confusion_matrix(y_val, y_test_pred)
# report = classification_report(y_val, y_test_pred)
# # print("Accuracy:", accuracy)
# print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean() * 100, scores.std() * 100))
# print("Confusion Matrix:")
# print(cm)
# print("Classification Report:")
# print(report)

"""###**Checking test and getting SVM output**"""

# svm = SVC(kernel='rbf', C=5, gamma='auto')

# pipeline = Pipeline([('scaler', StandardScaler()),
#                      #('pca', PCA(n_components=5)),
#                      ('nca', NeighborhoodComponentsAnalysis(n_components =n_nca)),
#                      ('svm', svm)])
# pipeline.fit(x_train,y_train)
# y_test_pred = pipeline.predict(test_scaled)

# # accuracy = accuracy_score(y_test_1, y_test_pred)
# # cm = confusion_matrix(y_test_1, y_test_pred)
# # report = classification_report(y_test_1, y_test_pred)
# # print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean() * 100, scores.std() * 100))
# # print("Confusion Matrix:")
# # print(cm)
# # print("Classification Report:")
# # print(report)

"""#**MLP**"""

# pipeline = Pipeline([('scaler', StandardScaler()),
#                      #('pca', PCA(n_components=n_pca)),
#                      ('nca', NeighborhoodComponentsAnalysis(n_components =n_nca)),
#                      ('MLP', MLPClassifier(random_state=42, max_iter=4))])
# pipeline.fit(x_train,y_train)
# y_test_pred = pipeline.predict(x_val)
# y_test_pred.shape
# accuracy = accuracy_score(y_val, y_test_pred)
# cm = confusion_matrix(y_val, y_test_pred)
# report = classification_report(y_val, y_test_pred)
# print("Accuracy:", accuracy)
# print("Confusion Matrix:")
# print(cm)
# print("Classification Report:")
# print(report)

"""###**Checking test and getting MLP output**"""

# pipeline = Pipeline([('scaler', StandardScaler()),
#                      #('pca', PCA(n_components=n_pca)),
#                      ('nca', NeighborhoodComponentsAnalysis(n_components =n_nca)),
#                      ('MLP', MLPClassifier(random_state=42, max_iter=20
#                      ))])
# pipeline.fit(x_train,y_train)
# # y_test_pred = pipeline.predict(x_test_1)
# # y_test_pred.shape
# # accuracy = accuracy_score(y_test_1, y_test_pred)
# # cm = confusion_matrix(y_test_1, y_test_pred)
# # report = classification_report(y_test_1, y_test_pred)
# # print("Accuracy:", accuracy)
# # print("Confusion Matrix:")
# # print(cm)
# # print("Classification Report:")
# # print(report)

"""#**KNN**"""

# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)
# print(neigh.predict([[1.1]]))
#                                               # predict(X)Predict the class labels for the provided data.
# score(X, y[, sample_weight])                  # Return the mean accuracy on the given test data and labels.
# print(neigh.predict_proba([[0.9]]))           # predict_proba(X) Return probability estimates for the test data X.

"""#**Random Forest**"""

# rf_classifier = RandomForestClassifier()
# rf_classifier.fit(x_train_selected, y_train)
# y_pred_rf = rf_classifier.predict(x_val_selected)

# accuracy_rf = accuracy_score(y_val, y_pred_rf)
# confusion_mat_rf = confusion_matrix(y_val, y_pred_rf)
# classification_rep_rf = classification_report(y_val, y_pred_rf)

# print(f"Accuracy (Random Forest): {accuracy_rf:.2f}")
# print(f"Confusion matrix (Random Forest):\n{confusion_mat_rf}")
# print(f"Classification report (Random Forest):\n{classification_rep_rf}")
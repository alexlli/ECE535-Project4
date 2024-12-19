import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
time_steps = 1000
features = 5

# Majority class
X_majority = np.random.randn(time_steps, features)
y_majority = np.zeros(time_steps)

# Minority class
X_minority = np.random.randn(int(time_steps * 0.1), features)
y_minority = np.ones(int(time_steps * 0.1))

# Combine into a single dataset
X = np.vstack((X_majority, X_minority))
y = np.concatenate((y_majority, y_minority))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Apply SMOTE for oversampling
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the class distribution
print("Original class distribution:", np.bincount(y_train.astype(int)))
print("Resampled class distribution:", np.bincount(y_train_resampled.astype(int)))

# Optionally, apply RandomUnderSampler to balance the dataset further
rus = RandomUnderSampler()
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_resampled, y_train_resampled)

# Check the new class distribution
print("Further resampled class distribution:", np.bincount(y_train_resampled.astype(int)))




import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=42)

# Step 1: Apply Z-score normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Rebalance the data using oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

print("Original class distribution:", np.bincount(y))
print("Resampled class distribution:", np.bincount(y_resampled))
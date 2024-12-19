import numpy as np
from scipy.stats import zscore

# Mock data to simulate mat_data dictionary
mat_data = {
    "x_train_modality_A": np.random.rand(100, 10),  # 100 samples, 10 features
    "x_train_modality_B": np.random.rand(100, 5),   # 100 samples, 5 features
    "y_train": np.random.randint(0, 2, size=(100, 1)), # 100 labels

    "x_test_modality_A": np.random.rand(20, 10),    # 20 samples, 10 features
    "x_test_modality_B": np.random.rand(20, 5),     # 20 samples, 5 features
    "y_test": np.random.randint(0, 2, size=(20, 1))   # 20 labels
}

# Modality identifiers
modality_A = "modality_A"
modality_B = "modality_B"

# Apply the provided code
data_train = {
    "A": zscore(mat_data[f"x_train_{modality_A}"]),
    "B": zscore(mat_data[f"x_train_{modality_B}"]),
    "y": np.squeeze(mat_data["y_train"])
}


data_test = {
    "A": zscore(mat_data[f"x_test_{modality_A}"]),
    "B": zscore(mat_data[f"x_test_{modality_B}"]),
    "y": np.squeeze(mat_data["y_test"])
}

# Checking the results
print("Training data (modality A) shape:", data_train["A"].shape)
print("Training data (modality B) shape:", data_train["B"].shape)
print("Training labels shape:", data_train["y"].shape)

print("Testing data (modality A) shape:", data_test["A"].shape)
print("Testing data (modality B) shape:", data_test["B"].shape)
print("Testing labels shape:", data_test["y"].shape)


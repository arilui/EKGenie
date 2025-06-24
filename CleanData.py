# ==================== DOWNLOAD DATA ====================
#imports
import pandas as pd 
import kagglehub as kh
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Download latest version
path = kh.dataset_download("shayanfazeli/heartbeat")

def load_data(path):
    mitbih_train = pd.read_csv(os.path.join(path, "mitbih_train.csv"), header=None)
    mitbih_test = pd.read_csv(os.path.join(path, "mitbih_test.csv"), header=None)
    ptbdb_normal = pd.read_csv(os.path.join(path, "ptbdb_normal.csv"), header=None)
    ptbdb_abnormal = pd.read_csv(os.path.join(path, "ptbdb_abnormal.csv"), header=None)
    return mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal

mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal = load_data(path)

# ==================== LABEL ASSIGNMENT ====================
def replace_label_column(data, label):
    data.iloc[:, -1] = label
    return data

ptbdb_normal = replace_label_column(ptbdb_normal, 6)  # normal
ptbdb_abnormal = replace_label_column(ptbdb_abnormal, 7)  # abnormal

ptbdb_all = pd.concat([ptbdb_normal, ptbdb_abnormal], ignore_index=True)

# ==================== SPLIT DATA ====================
X_ptbdb_train, X_ptbdb_test, y_ptbdb_train, y_ptbdb_test = train_test_split(
    ptbdb_all.iloc[:, :-1], ptbdb_all.iloc[:, -1], test_size=0.2, random_state=42
)

X_train = pd.concat([X_ptbdb_train, mitbih_train.iloc[:, :-1]], ignore_index=True)
X_test = pd.concat([X_ptbdb_test, mitbih_test.iloc[:, :-1]], ignore_index=True)

y_train = pd.concat([y_ptbdb_train, mitbih_train.iloc[:, -1]], ignore_index=True)
y_test = pd.concat([y_ptbdb_test, mitbih_test.iloc[:, -1]], ignore_index=True)

# ==================== LABEL REMAPPING ====================
# Merge class 6 into 0 ("Normal"), rename 7 to 5 ("Abnormal")
y_train = pd.Series(y_train).astype(int).replace({6: 0, 7: 5}).reset_index(drop=True)
y_test = pd.Series(y_test).astype(int).replace({6: 0, 7: 5}).reset_index(drop=True)

# ==================== DROP LOW-INFORMATION COLUMNS ====================
def drop_almost_zero_columns(X, threshold=0.8):
    zero_fraction = (X == 0).sum() / len(X)
    drop_cols = zero_fraction[zero_fraction >= threshold].index
    return X.drop(columns=drop_cols)

X_train = drop_almost_zero_columns(X_train)
X_test = drop_almost_zero_columns(X_test)

# ==================== CLASS COUNT ====================
unique_classes = np.unique(np.concatenate([y_train, y_test]))
num_classes = len(unique_classes)

# ==================== CLASS WEIGHTS ====================
class_weights = {
    0: 1.0,   # Normal
    1: 4.0,   # Supraventricular
    2: 1.5,   # Ventricular
    3: 5.0,   # Fusion
    4: 1.0,   # Unknown
    5: 1.2    # Abnormal
}
conditions = {
    0: "Normal (N)",
    1: "Supraventricular premature (S)",
    2: "Premature ventricular contraction (V)",
    3: "Fusion of ventricular and normal beat (F)",
    4: "Unclassifiable beat (Q)",
    5: "Abnormal (A)"
}

def get_clean_data():
    """
    Returns the cleaned training and testing datasets.
    """
    return X_train, X_test, y_train, y_test, class_weights, conditions
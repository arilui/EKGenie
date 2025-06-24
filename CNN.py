import CleanData
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ==================== DATA PREP ====================

# Import data and convert to numpy arrays
X_train, X_test, y_train, y_test, class_weights, conditions = CleanData.get_clean_data()
num_classes = CleanData.num_classes
X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

# Normalize (if not done in CleanData)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# ==================== FOCAL LOSS DEFINITION ====================

from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K

@tf.keras.utils.register_keras_serializable()
class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.5, reduction='sum_over_batch_size', name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # Clip to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate focal weight
        weight = self.alpha * K.pow(1. - y_pred, self.gamma)
        
        # Calculate loss
        loss = weight * cross_entropy
        
        # Sum over classes
        return K.sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config

if __name__ == "__main__":
    # ==================== BUILD CNN ====================
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=FocalLoss(gamma=2.0, alpha=0.5),
        metrics=['accuracy']
    )
    model.summary()

    # ==================== TRAIN MODEL ====================
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train_cat,
        epochs=30,
        batch_size=128,
        validation_data=(X_test, y_test_cat),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    model.save('cnn_model.keras')

    # ==================== EVALUATION ====================
    # Add your evaluation code here if needed
    
    print('Training completed successfully!')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import List
from ..core.interfaces import AlphaModel
from ..core.types import Insight
import tensorflow.keras.backend as K

class AchillesLSTM(AlphaModel):
    def __init__(self, input_shape, name="Achilles_Antigravity_LSTM"):
        """
        Initializes the Antigravity LSTM model (Bi-LSTM + Focal Loss).
        """
        super().__init__(name=name)
        self.input_shape = input_shape
        self.model = self._build_model()

    def categorical_focal_loss(self, gamma=2.0, alpha=0.25):
        """
        Softmax Focal Loss for Multi-class Classification.
        Punishes the model for ignoring minority classes (Buy/Sell).
        """
        def focal_loss_fn(y_true, y_pred):
            # Clip for stability
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            
            # Cross Entropy
            cross_entropy = -y_true * K.log(y_pred)
            
            # Focal Weighting
            weight = alpha * y_true * K.pow((1 - y_pred), gamma)
            
            # Final Loss
            loss = weight * cross_entropy
            return K.sum(loss, axis=1)
        return focal_loss_fn

    def update(self, data) -> List[Insight]:
        """
        Predicts signals based on new data.
        Returns a list of Insight objects.
        """
        # In production, 'data' is the input tensor (1, 60, 4)
        if hasattr(data, 'shape'):
             prediction = self.model.predict(data, verbose=0)
             # Parse prediction (Buy, Sell, Hold)...
             # For now returning empty as this is just the model structure
             pass
        return []

    def _build_model(self):
        model = Sequential()
        
        # [TAG: ANTIGRAVITY_ARCHITECTURE]
        # 1. Bi-Directional LSTM Layer 1
        # Reads sequence Past->Future AND Future->Past (conceptually in window)
        model.add(Input(shape=self.input_shape))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        
        # 2. Bi-Directional LSTM Layer 2
        model.add(Bidirectional(LSTM(32, return_sequences=False)))
        model.add(BatchNormalization()) # Stabilize internal covariance shift
        model.add(Dropout(0.3))
        
        # 3. Output Layer
        model.add(Dense(3, activation='softmax'))
        
        # Optimizer: AdamW
        try:
            from tensorflow.keras.optimizers import AdamW
        except ImportError:
            from tensorflow.keras.optimizers.experimental import AdamW
            
        optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
        
        # Compile with Focal Loss
        model.compile(
            optimizer=optimizer, 
            loss=self.categorical_focal_loss(gamma=2.0, alpha=0.25), 
            metrics=['accuracy']
        )
        
        return model

    def train(self, x_train, y_train, epochs=100, batch_size=128, validation_split=0.15):
        """
        Antigravity Training Loop
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
    def save(self, path="achilles_antigravity.keras"):
        self.model.save(path)
        print(f"Model saved to {path}")


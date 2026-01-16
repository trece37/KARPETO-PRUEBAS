"""
AchillesLSTM Titan V3
Architecture: Bi-LSTM + Attention Mechanism
Loss: Categorical Focal Loss
Optimizer: AdamW (Weight Decay)
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization, Multiply, Permute, Flatten
import tensorflow.keras.backend as K
from typing import List
import os

# Manteniendo referencias relativas a la estructura existente
from ..core.interfaces import AlphaModel
from ..core.types import Insight, InsightType, InsightDirection

class AchillesLSTM(AlphaModel):
    def __init__(self, input_shape, name="Achilles_LSTM_V3_Titan"):
        super().__init__(name=name)
        self.input_shape = input_shape
        self.model = self._build_model()
        self.model_path = "models/achilles_lstm_v3_titan.keras" # Ruta relativa local

    def categorical_focal_loss(self, gamma=2.0, alpha=0.25):
        """
        Función de pérdida para clases desbalanceadas (Trading).
        Penaliza los errores en Buy/Sell (difíciles) más que en Hold (fácil).
        """
        def focal_loss_fn(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            cross_entropy = -y_true * K.log(y_pred)
            weight = alpha * y_true * K.pow((1 - y_pred), gamma)
            loss = weight * cross_entropy
            return K.sum(loss, axis=1)
        return focal_loss_fn

    def _attention_block(self, inputs, time_steps):
        """
        Mecanismo de atención para enfocar minutos clave.
        """
        a = Permute((2, 1))(inputs)
        a = Dense(time_steps, activation='softmax')(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        
        # Bi-LSTM Layer 1
        lstm_out = Bidirectional(LSTM(units=128, return_sequences=True))(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Bi-LSTM Layer 2
        lstm_out = LSTM(units=64, return_sequences=True)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Attention Layer
        attention_mul = self._attention_block(lstm_out, self.input_shape[0])
        attention_mul = Flatten()(attention_mul)
        
        # Dense Decision Layer
        dense = Dense(64, activation='relu')(attention_mul)
        dense = Dropout(0.2)(dense)
        
        # Output Layer (3 Classes: Hold, Buy, Sell)
        outputs = Dense(units=3, activation='softmax')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # AdamW Optimizer (R3K Standard)
        try:
            from tensorflow.keras.optimizers import AdamW
        except ImportError:
            from tensorflow.keras.optimizers.experimental import AdamW
            
        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
        
        model.compile(optimizer=optimizer, 
                      loss=self.categorical_focal_loss(gamma=2.0, alpha=0.25), 
                      metrics=['accuracy'])
        return model

    def update(self, input_tensor) -> List[Insight]:
        """Inferencia en tiempo real."""
        if not os.path.exists(self.model_path):
             print("⚠️ MODEL NOT FOUND. RETURNING EMPTY INSIGHTS.")
             return []

        prediction = self.model.predict(input_tensor, verbose=0)
        signal_idx = prediction.argmax(axis=-1)[0]
        confidence = prediction.max(axis=-1)[0]
        
        direction = InsightDirection.FLAT
        if signal_idx == 1: direction = InsightDirection.UP
        if signal_idx == 2: direction = InsightDirection.DOWN
        
        insight = Insight(
            symbol="XAUUSD", 
            generated_time_utc=None,
            type=InsightType.PRICE,
            direction=direction,
            period=None,
            confidence=float(confidence)
        )
        return [insight]

    def load_weights(self, path=None):
        p = path if path else self.model_path
        if os.path.exists(p):
            self.model.load_weights(p)
            print(f"✅ LSTM Weights Loaded: {p}")
        else:
            print(f"⚠️ Weights not found at {p}. Model untrained.")

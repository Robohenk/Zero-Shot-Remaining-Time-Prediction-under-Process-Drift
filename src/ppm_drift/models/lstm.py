from __future__ import annotations

from .base import BaseRemainingTimeModel


class LSTMRegressor(BaseRemainingTimeModel):
    def build(self):
        import tensorflow as tf

        hidden_dim = int(self.kwargs.get("hidden_dim", 64))
        dropout = float(self.kwargs.get("dropout", 0.1))
        learning_rate = float(self.kwargs.get("learning_rate", 1e-3))
        inp = tf.keras.Input(shape=(self.encoding.max_len, self.encoding.feature_dim), name="x")
        x = tf.keras.layers.Masking(mask_value=0.0)(inp)
        x = tf.keras.layers.LSTM(hidden_dim, dropout=dropout)(x)
        out = tf.keras.layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs=inp, outputs=out, name="lstm_regressor")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mae",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        return model

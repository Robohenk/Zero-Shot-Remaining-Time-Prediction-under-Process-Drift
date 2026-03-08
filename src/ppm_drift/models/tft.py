from __future__ import annotations

from .base import BaseRemainingTimeModel


class TemporalFusionTransformerRegressor(BaseRemainingTimeModel):
    """Compact TFT-style regressor.

    This is a pragmatic Keras implementation inspired by TFT building blocks:
    variable projection, gated residual blocks, recurrent context, self-attention,
    and quantile-free point regression. It is suitable as a strong comparison model,
    but it is still more lightweight than the full canonical TFT implementation.
    """

    def build(self):
        import tensorflow as tf

        d_model = int(self.kwargs.get("hidden_dim", 64))
        num_heads = int(self.kwargs.get("transformer_heads", 4))
        ff_dim = int(self.kwargs.get("transformer_ff_dim", 128))
        dropout = float(self.kwargs.get("dropout", 0.1))
        learning_rate = float(self.kwargs.get("learning_rate", 1e-3))

        def gated_residual(x, units):
            skip = tf.keras.layers.Dense(units)(x)
            z = tf.keras.layers.Dense(units, activation="elu")(x)
            z = tf.keras.layers.Dropout(dropout)(z)
            z = tf.keras.layers.Dense(units)(z)
            gate = tf.keras.layers.Dense(units, activation="sigmoid")(x)
            out = gate * z + (1.0 - gate) * skip
            return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out)

        inp = tf.keras.Input(shape=(self.encoding.max_len, self.encoding.feature_dim), name="x")
        x = tf.keras.layers.Masking(mask_value=0.0)(inp)
        x = tf.keras.layers.Dense(d_model)(x)
        x = gated_residual(x, d_model)
        x = tf.keras.layers.LSTM(d_model, return_sequences=True)(x)
        attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(d_model // num_heads, 1), dropout=dropout)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        x = gated_residual(x, d_model)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(ff_dim, activation="relu"))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(d_model, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        out = tf.keras.layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inp, out, name="compact_tft_regressor")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mae",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        return model

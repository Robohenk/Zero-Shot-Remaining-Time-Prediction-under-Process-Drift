from __future__ import annotations

from .base import BaseRemainingTimeModel


class TransformerEncoderRegressor(BaseRemainingTimeModel):
    def build(self):
        import tensorflow as tf

        d_model = int(self.kwargs.get("hidden_dim", 64))
        num_heads = int(self.kwargs.get("transformer_heads", 4))
        ff_dim = int(self.kwargs.get("transformer_ff_dim", 128))
        n_layers = int(self.kwargs.get("transformer_layers", 2))
        dropout = float(self.kwargs.get("dropout", 0.1))
        learning_rate = float(self.kwargs.get("learning_rate", 1e-3))

        inp = tf.keras.Input(shape=(self.encoding.max_len, self.encoding.feature_dim), name="x")
        x = tf.keras.layers.Masking(mask_value=0.0)(inp)
        x = tf.keras.layers.Dense(d_model)(x)

        positions = tf.range(start=0, limit=self.encoding.max_len, delta=1)
        pos_emb = tf.keras.layers.Embedding(input_dim=self.encoding.max_len, output_dim=d_model)(positions)
        x = x + pos_emb

        for _ in range(n_layers):
            attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(d_model // num_heads, 1), dropout=dropout)(x, x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
            ff = tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
            ])(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        out = tf.keras.layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inp, out, name="transformer_encoder_regressor")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mae",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        return model

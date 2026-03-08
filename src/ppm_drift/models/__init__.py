from .lstm import LSTMRegressor
from .transformer import TransformerEncoderRegressor
from .tft import TemporalFusionTransformerRegressor


def build_model(model_name: str, encoding, **kwargs):
    model_name = model_name.lower()
    if model_name == "lstm":
        return LSTMRegressor(encoding, **kwargs)
    if model_name in {"transformer", "encoder_transformer"}:
        return TransformerEncoderRegressor(encoding, **kwargs)
    if model_name == "tft":
        return TemporalFusionTransformerRegressor(encoding, **kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")

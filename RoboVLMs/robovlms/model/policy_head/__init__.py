from .base_policy import LSTMDecoder, FCDecoder, DiscreteDecoder, GPTDecoder

# MobileVLALSTMDecoder is an alias for LSTMDecoder
class MobileVLALSTMDecoder(LSTMDecoder):
    """Mobile VLA LSTM Decoder - alias for LSTMDecoder with mobile-specific configurations"""
    pass

__all__ = ["LSTMDecoder", "FCDecoder", "DiscreteDecoder", "GPTDecoder", "MobileVLALSTMDecoder"]

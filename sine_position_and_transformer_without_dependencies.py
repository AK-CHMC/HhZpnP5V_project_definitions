import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as tfl

# assumes mask is passed to layer in input, but shouldn't be strictly necessary - untested though
# implementation comes from https://github.com/keras-team/keras-nlp/blob/v0.4.1/keras_nlp/layers/transformer_encoder.py
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tfl.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tfl.Dense(ff_dim, activation="relu"), tfl.Dense(embed_dim)])
        self.layernorm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(rate)
        self.dropout2 = tfl.Dropout(rate)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    def call(self, inputs,mask=None,training=True):
        if mask is not None:
            encoder_pad_mask = tf.math.not_equal(mask, 0)  # shape [B, S]
            attention_mask = encoder_pad_mask[:, tf.newaxis]
        else:
            attention_mask = None        
        attn_output = self.att(inputs, inputs,attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# defines SinePositionEncoding from https://github.com/keras-team/keras-nlp/blob/v0.4.1/keras_nlp/layers/sine_position_encoding.py
class SinePositionEncoding(Layer):
    def __init__(self,max_wavelength=10000,**kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        # length of sequence is the second last dimension of the inputs
        seq_length = input_shape[-2]
        hidden_size = input_shape[-1]
        position = tf.cast(tf.range(seq_length), self.compute_dtype)
        min_freq = tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
        timescales = tf.pow(
            min_freq,
            tf.cast(2 * (tf.range(hidden_size) // 2), self.compute_dtype)
            / tf.cast(hidden_size, self.compute_dtype),
        )
        angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(hidden_size) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask)

        return tf.broadcast_to(positional_encodings, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"max_wavelength": self.max_wavelength})
        return config

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# ==========================================
# 1. UTILIDADES CORREGIDAS (Máscaras Booleanas)
# ==========================================

def create_padding_mask(seq):
    # Devuelve True donde NO es padding (queremos conservar)
    # False donde ES padding (queremos ignorar)
    # Nota: Keras 3 usa True=Keep, False=Drop en attention_mask
    seq = tf.cast(tf.math.not_equal(seq, 0), tf.bool)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    # Máscara Triangular Inferior (Lo que ya pasó)
    # Devuelve 1s (True) en el pasado y presente, 0s (False) en el futuro
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # Convertimos a Booleano: True = Permitir atención
    return tf.cast(mask, tf.bool) 

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # 1. Definimos las capas (SIN return_attention_scores aquí)
        self.mha1 = layers.MultiHeadAttention(num_heads, key_dim=d_model)
        self.mha2 = layers.MultiHeadAttention(num_heads, key_dim=d_model)

        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # 1. Auto-Atención (mha1)
        # No pedimos scores, así que devuelve 1 valor (el tensor)
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 2. Atención Cruzada (mha2)
        # ¡AQUÍ es donde pedimos los scores! (En el método call)
        attn2, attn_weights = self.mha2(
            out1, enc_output, enc_output, 
            attention_mask=padding_mask, 
            return_attention_scores=True, # <--- Argumento movido aquí
            training=training
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # 3. Feed Forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights

class TransformerDecoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block = self.dec_layers[i](
                x, 
                enc_output=enc_output, 
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )
            attention_weights[f'decoder_layer{i+1}_block2'] = block

        x = self.final_layer(x)
        return x, attention_weights

import tensorflow as tf
from tensorflow.keras import layers

# -------------------------------------------------------
# Bahdanau Attention
# -------------------------------------------------------
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch, 49, embed_dim)
        # hidden:   (batch, units)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# -------------------------------------------------------
# Decoder corregido (sin concatenación antes del GRU)
# -------------------------------------------------------
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding_dim = embedding_dim

        self.embedding = layers.Embedding(vocab_size, embedding_dim)

        self.gru = layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

        # Densas finales
        self.fc1 = layers.Dense(self.units)
        self.fc2 = layers.Dense(vocab_size)

        # Atención
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        """
        x        : (batch, 1) — token previo
        features : (batch, 49, embed_dim) — salida del encoder
        hidden   : (batch, units)
        """

        # 1. Atención basada en hidden
        context_vector, attention_weights = self.attention(features, hidden)

        # 2. Embedding del token actual
        x = self.embedding(x)  # (batch, 1, embedding_dim)

        # 3. Pasar el embedding SOLO por la GRU
        output, state = self.gru(x, initial_state=hidden)
        # output: (batch, 1, units)

        # 4. Concatenar salida del GRU con contexto
        x = tf.concat([output, tf.expand_dims(context_vector, 1)], axis=-1)
        # x: (batch, 1, units + units)

        # 5. Clasificación
        x = self.fc1(x)
        x = tf.reshape(x, (-1, x.shape[2]))  # (batch, units)
        x = self.fc2(x)                      # (batch, vocab_size)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

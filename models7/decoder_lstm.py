import tensorflow as tf
from tensorflow.keras import layers

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        # features (Encoder output): (batch_size, 49, embedding_dim)
        # hidden (Decoder hidden state): (batch_size, hidden_size)
        
        # Expandimos hidden para que tenga dimensión de tiempo: (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # Calculamos el score de atención
        # score shape: (batch_size, 49, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

        # Pesos de atención (softmax para que sumen 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Vector de contexto (suma ponderada de las características de la imagen)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        # Embedding para las palabras de entrada (input al decoder)
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        # GRU (Gated Recurrent Unit) - versión simplificada de LSTM
        self.gru = layers.GRU(self.units,
                              return_sequences=True,
                              return_state=True,
                              recurrent_initializer='glorot_uniform')
        
        self.fc1 = layers.Dense(self.units)
        self.fc2 = layers.Dense(vocab_size) # Salida: probabilidad de cada palabra del vocabulario

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # x: input actual (palabra anterior), shape (batch_size, 1)
        
        # 1. Atención: Mirar a la imagen basándose en el estado oculto anterior
        context_vector, attention_weights = self.attention(features, hidden)

        # 2. Embedding de la palabra entrada
        x = self.embedding(x) # (batch_size, 1, embedding_dim)

        # 3. Concatenar vector de contexto (imagen) con el embedding (palabra)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 4. Pasar por la GRU
        output, state = self.gru(x)

        # 5. Clasificación final
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x) # (batch_size, vocab_size)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

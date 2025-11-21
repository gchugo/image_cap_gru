import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers

class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim=256, trainable_resnet=False):
        """
        Args:
            embedding_dim (int): Tamaño del vector al que queremos reducir la imagen (ej. 256).
            trainable_resnet (bool): Si queremos re-entrenar las capas de ResNet (Fine-tuning).
                                     Por defecto False para congelar los pesos de ImageNet.
        """
        super(Encoder, self).__init__()
        
        # 1. Cargar ResNet50 pre-entrenada
        # include_top=False: Quitamos la capa de clasificación final (las 1000 clases de ImageNet).
        # weights='imagenet': Usamos los pesos aprendidos.
        # pooling=None: IMPORTANTE mantenerlo en None para conservar la estructura espacial (7x7) para la Atención.
        self.resnet = ResNet50(include_top=False, weights='imagenet', pooling=None)
        
        # Congelar los pesos de la ResNet para no dañarlos al inicio del entrenamiento
        self.resnet.trainable = trainable_resnet
        
        # 2. Capa Densa para adaptar la salida al tamaño que queramos (embedding_dim)
        self.fc = layers.Dense(embedding_dim, activation='relu')

    def call(self, x, training=False):
        # x shape de entrada: (batch_size, 224, 224, 3)
        
        # Pasamos por ResNet (training=False mantiene BatchNorm en modo inferencia)
        x = self.resnet(x, training=training) 
        # Salida de ResNet: (batch_size, 7, 7, 2048)
        
        # Aplanamos las dimensiones espaciales (7x7 -> 49) para tener una secuencia de características
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3])) 
        # Ahora shape: (batch_size, 49, 2048)
        
        # Pasamos por la capa densa para reducir dimensiones (2048 -> 256)
        features = self.fc(x)
        # Shape final: (batch_size, 49, embedding_dim)
        
        return features
import tensorflow as tf
from models.decoder_transformer import create_padding_mask, create_look_ahead_mask


class Trainer:
    def __init__(self, encoder, decoder, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function  # Ahora SÍ funciona bien aquí
    def train_step(self, img_tensor, target, start_token_id):
        loss = 0
        
        # Inicializar estado oculto
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        # Input inicial para el decoder: <start>
        dec_input = tf.expand_dims([start_token_id] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            # Usamos self.encoder
            features = self.encoder(img_tensor, training=True)

            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])
        
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return total_loss
    
    @tf.function
    def validate_step(self, img_tensor, target, start_token_id):
        """
        Calcula el loss de validación sin actualizar pesos.
        """
        loss = 0
        
        # Inicializar estado oculto
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        # Input inicial
        dec_input = tf.expand_dims([start_token_id] * target.shape[0], 1)

        # A. Encoder (training=False es importante para BatchNorm/Dropout)
        features = self.encoder(img_tensor, training=False)

        # B. Bucle Decoder
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            loss += self.loss_function(target[:, i], predictions)
            
            # En validación usamos Teacher Forcing igual para calcular el Loss correctamente
            dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])
        return total_loss

def get_optimizer(learning_rate=1e-4):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


# 1. Learning Rate Schedule (OBLIGATORIO para convergencia de Transformers)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# 2. Nueva Clase Trainer Específica para Transformer
class TransformerTrainer:
    def __init__(self, encoder, decoder, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def create_masks_decoder(self, tar):
        # Asegúrate de tener importadas create_look_ahead_mask y create_padding_mask
        # O defínelas dentro de este archivo o impórtalas de models.decoder_transformer
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return combined_mask

    # NOTA: Si estás en local y te daba error de grafo, prueba sin @tf.function primero.
    # Si quieres velocidad, descoméntalo, pero Keras 3 a veces da guerra con @tf.function manual.
    # @tf.function 
    def train_step(self, img_tensor, target):
        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        combined_mask = self.create_masks_decoder(tar_inp)

        with tf.GradientTape() as tape:
            # Llamada al Encoder
            img_features = self.encoder(img_tensor, training=True)
            
            # Llamada al Decoder (CORREGIDA PARA KERAS 3)
            # Pasamos argumentos por nombre
            predictions, _ = self.decoder(
                tar_inp, 
                enc_output=img_features,    # <--- Keyword argument
                training=True,              # <--- Keyword argument
                look_ahead_mask=combined_mask, # <--- Keyword argument
                padding_mask=None           # <--- Keyword argument
            )

            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))

        return loss
    
    # @tf.function
    def validate_step(self, img_tensor, target):
        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]
        combined_mask = self.create_masks_decoder(tar_inp)

        img_features = self.encoder(img_tensor, training=False)
        
        # Llamada al Decoder (CORREGIDA PARA KERAS 3)
        predictions, _ = self.decoder(
            tar_inp, 
            enc_output=img_features,    # <--- Keyword argument
            training=False,             # <--- Keyword argument
            look_ahead_mask=combined_mask, # <--- Keyword argument
            padding_mask=None           # <--- Keyword argument
        )
        
        loss = self.loss_function(tar_real, predictions)
        return loss

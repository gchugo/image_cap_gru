import tensorflow as tf
import os
import csv
import re
from collections import defaultdict
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import random


# ==========================================
# 1. Funciones de Carga y Limpieza de Texto
# ==========================================

def load_captions_csv(path):
    """
    Lee el archivo captions.txt y devuelve un diccionario {img_name: [caption1, caption2...]}
    """
    captions = defaultdict(list)
    
    # Verificamos si el archivo existe
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de captions en: {path}")

    with open(path, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # saltar encabezado "image,caption"
        
        for row in reader:
            if len(row) < 2:
                continue
            img_name = row[0].strip()
            caption = row[1].strip()
            captions[img_name].append(caption)
    
    return captions

def clean_caption(caption):
    """
    Limpia y normaliza el texto de un caption.
    """
    caption = caption.lower()
    caption = re.sub(r"[^a-zA-Z0-9]+", " ", caption)  # quitar todo lo que no sea alfanumérico
    caption = re.sub(r"\s+", " ", caption).strip()     # normalizar espacios
    return caption

# ==========================================
# 2. Funciones de Procesamiento de Imágenes
# ==========================================

def _load_and_preprocess_py(img_name_tensor, caption_tensor, images_dir):
    """
    Función interna (Python puro) ejecutada por tf.py_function.
    Carga la imagen desde disco y la preprocesa para ResNet50.
    """
    try:
        # 1. Decodificar el nombre de la imagen (bytes -> string)
        img_name_str = img_name_tensor.numpy().decode('utf-8')
        
        # 2. Construir ruta completa usando el argumento images_dir
        img_path = os.path.join(images_dir, img_name_str)
        
        # 3. Cargar, convertir a array y preprocesar
        # target_size=(224, 224) es obligatorio para ResNet50
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        
        # Preprocesamiento específico de ResNet (resta la media de ImageNet, etc.)
        img_array = preprocess_input(img_array)
        
        return img_array, caption_tensor.numpy()
        
    except Exception as e:
        # Es útil imprimir el error si falla la carga de una imagen específica
        print(f"Error cargando imagen {img_name_tensor}: {e}")
        # Devolvemos ceros en caso de error para no romper el pipeline (opcional)
        return params_error_handling(224) 

def params_error_handling(size):
    return tf.zeros((size, size, 3)), tf.zeros((1,), dtype=tf.int32)


# ==========================================
# 3. Función Principal para crear el Dataset
# ==========================================

def create_tf_dataset(img_paths, seqs, images_dir, max_len, batch_size=64, buffer_size=1000):
    """
    Crea y devuelve un objeto tf.data.Dataset listo para entrenar.
    
    Args:
        img_paths (list): Lista de nombres de archivos de imágenes.
        seqs (list): Lista de secuencias tokenizadas y paddeadas.
        images_dir (str): Ruta a la carpeta de imágenes.
        max_len (int): Longitud máxima de la secuencia (para set_shape).
        batch_size (int): Tamaño del batch.
        buffer_size (int): Tamaño del buffer para shuffle.
    """
    
    # Definimos el wrapper aquí dentro para capturar 'images_dir' y 'max_len'
    # sin usar variables globales.
    def tf_map_wrapper(img_name, caption):
        img, cap = tf.py_function(
            # Usamos una lambda para pasar images_dir a la función python
            func=lambda i, c: _load_and_preprocess_py(i, c, images_dir),
            inp=[img_name, caption],
            Tout=[tf.float32, tf.int32]
        )
        
        # ¡IMPORTANTE! Restaurar las formas (shapes)
        img.set_shape([224, 224, 3])
        cap.set_shape([max_len]) 
        
        return img, cap

    # 1. Crear dataset de slices
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, seqs))
    
    # 2. Aplicar el mapeo (carga de imágenes) en paralelo
    dataset = dataset.map(tf_map_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 3. Shuffle, Batch y Prefetch
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# ==========================================
# 4. Funciones de División y Utilidades
# ==========================================

def split_dataset(captions_dict, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Divide las imágenes únicas en conjuntos de entrenamiento, validación y prueba.
    Esto previene el 'data leakage' asegurando que todas las descripciones de una
    misma imagen permanezcan en el mismo conjunto.
    
    Args:
        captions_dict (dict): Diccionario {img_name: [cap1, cap2...]}
        train_ratio (float): Proporción para entrenamiento (ej. 0.8).
        val_ratio (float): Proporción para validación (ej. 0.1).
        seed (int): Semilla para reproducibilidad.
        
    Returns:
        tuple: Tres listas con los nombres de las imágenes (train_imgs, val_imgs, test_imgs).
    """
    # Obtener lista de nombres de imágenes únicas
    all_img_names = list(captions_dict.keys())
    
    # Mezclar aleatoriamente
    if seed is not None:
        random.seed(seed)
    random.shuffle(all_img_names)
    
    # Calcular puntos de corte
    n = len(all_img_names)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Dividir
    train_imgs = all_img_names[:train_end]
    val_imgs = all_img_names[train_end:val_end]
    test_imgs = all_img_names[val_end:]
    
    print(f"Split realizado: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
    return train_imgs, val_imgs, test_imgs

def flatten_dataset_data(img_names, captions_dict):
    """
    Convierte una lista de nombres de imágenes y el diccionario maestro en dos listas paralelas
    (img_paths, captions) listas para ser usadas en el entrenamiento/tokenización.
    
    Args:
        img_names (list): Lista de nombres de imágenes (ej. output de split_dataset).
        captions_dict (dict): Diccionario maestro {img_name: [cap1, cap2...]}.
        
    Returns:
        tuple: (imgs_list, caps_list) donde imgs_list repite el nombre de la imagen
               por cada caption que tenga.
    """
    imgs_list = []
    caps_list = []
    
    for img in img_names:
        caption_list = captions_dict[img]
        for cap in caption_list:
            imgs_list.append(img)
            caps_list.append(cap)
            
    return imgs_list, caps_list

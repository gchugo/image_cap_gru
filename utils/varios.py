import os
from PIL import Image
import matplotlib.pyplot as plt

def show_image_all_captions(img_name, captions_dict, images_dir):
    img_path = os.path.join(images_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
    print("Captions de", img_name)
    for c in captions_dict[img_name]:
        print(" •", c)

import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_training_curves(train_loss, val_loss, bleu_scores=None, title="Rendimiento del Modelo", save_path=None):
    """
    Pinta dos gráficas:
    1. Loss de Entrenamiento vs Validación.
    2. (Opcional) Evolución del BLEU Score.
    """
    epochs = range(1, len(train_loss) + 1)
    
    # Si hay datos de BLEU, hacemos 2 subplots
    if bleu_scores and len(bleu_scores) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfica 1: LOSS
        ax1.plot(epochs, train_loss, 'b-o', label='Train Loss')
        ax1.plot(epochs, val_loss, 'r-o', label='Val Loss')
        ax1.set_title('Curvas de Loss')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfica 2: BLEU
        # Asumimos que bleu_scores tiene la misma longitud o ajustamos el eje x
        bleu_epochs = range(1, len(bleu_scores) + 1)
        ax2.plot(bleu_epochs, bleu_scores, 'g-s', label='Val BLEU-1')
        ax2.set_title('Evolución de Calidad (BLEU)')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('BLEU Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    else:
        # Solo gráfica de Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
        plt.plot(epochs, val_loss, 'r-o', label='Val Loss')
        plt.title(title)
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfica guardada en {save_path}")
    
    plt.show()

def save_history_json(history_dict, path='training_history.json'):
    """Guarda las listas de loss/bleu en un JSON para no perderlas."""
    # Convertir numpy types a float nativo de python
    serializable = {}
    for k, v in history_dict.items():
        serializable[k] = [float(x) for x in v]
        
    with open(path, 'w') as f:
        json.dump(serializable, f)

def load_history_json(path='training_history.json'):
    """Carga el historial previo."""
    if not os.path.exists(path):
        return {'train_loss': [], 'val_loss': [], 'val_bleu': []}
        
    with open(path, 'r') as f:
        return json.load(f)

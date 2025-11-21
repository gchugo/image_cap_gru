import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from tqdm.notebook import tqdm # Importamos la versión bonita para notebooks
import os
from tqdm.notebook import tqdm
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

def load_image_for_eval(image_path):
    """Carga y preprocesa una imagen individual para inferencia."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, image_path

def greedy_evaluate(image_path, encoder, decoder, tokenizer, max_len):
    """
    Genera un caption usando Greedy Search (Argmax).
    Más rápido que Beam Search, pero a veces menos preciso.
    """
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']

    # 1. Procesar imagen
    img = load_image_for_eval(image_path)
    img = tf.expand_dims(img, 0) 
    
    # 2. Encoder
    features = encoder(img, training=False)

    # 3. Inicializar Decoder
    hidden = decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([start_token], 0)

    result_ids = []

    # 4. Bucle de generación
    for i in range(max_len):
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        
        # Elegir la palabra con mayor probabilidad (GREEDY)
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        if predicted_id == end_token:
            break
            
        result_ids.append(predicted_id)
        
        # La predicción es la entrada del siguiente paso
        dec_input = tf.expand_dims([predicted_id], 0)

    # Decodificar a texto
    result_caption = [tokenizer.index_word[i] for i in result_ids]
    return ' '.join(result_caption)

def calculate_metrics_greedy(encoder, decoder, tokenizer, max_len, test_paths, all_captions_dict, sample_size=None):
    """
    Calcula BLEU, METEOR y ROUGE-L usando Greedy Search para GRU.
    """
    scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    actual_tokens = []
    predicted_tokens = []
    meteor_scores = []
    rouge_scores = []
    
    eval_paths = test_paths[:sample_size] if sample_size else test_paths
        
    for img_path in tqdm(eval_paths):
        # 1. Generar Predicción (GREEDY)
        pred_str = greedy_evaluate_gru(
            img_path, encoder, decoder, tokenizer, max_len
        )
        pred_toks = pred_str.split()
        
        # 2. Obtener Referencias
        img_name = os.path.basename(img_path)
        raw_captions = all_captions_dict.get(img_name, [])
        
        ref_list_tokens = []
        ref_list_strs = []
        
        for c in raw_captions:
            c_clean = c.replace('<start>', '').replace('<end>', '').strip()
            ref_list_strs.append(c_clean)
            ref_list_tokens.append(c_clean.split())
            
        # Acumular
        actual_tokens.append(ref_list_tokens)
        predicted_tokens.append(pred_toks)
        
        # METEOR
        m_score = meteor_score(ref_list_tokens, pred_toks)
        meteor_scores.append(m_score)
        
        # ROUGE-L
        best_rouge = 0
        for ref in ref_list_strs:
            scores = scorer_rouge.score(ref, pred_str)
            if scores['rougeL'].fmeasure > best_rouge:
                best_rouge = scores['rougeL'].fmeasure
        rouge_scores.append(best_rouge)
    
    b1 = corpus_bleu(actual_tokens, predicted_tokens, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    avg_meteor = np.mean(meteor_scores)
    avg_rouge = np.mean(rouge_scores)
    
    print(f"BLEU-1:  {b1:.4f}")
    print(f"BLEU-2:  {b2:.4f}")
    print(f"BLEU-3:  {b3:.4f}")
    print(f"BLEU-4:  {b4:.4f}")
    print(f"METEOR:  {avg_meteor:.4f}")
    print(f"ROUGE-L: {avg_rouge:.4f}")
    
    return {'bleu1': b1, 'bleu4': b4, 'meteor': avg_meteor, 'rouge': avg_rouge}

ef calculate_bleu_score(encoder, decoder, tokenizer, max_len, test_img_paths, all_captions_dict, sample_size=None):
    """
    Calcula los scores BLEU-1 a BLEU-4 para el conjunto de prueba.
    
    Args:
        encoder, decoder: Modelos entrenados.
        tokenizer: Tokenizador ajustado.
        max_len: Longitud máxima de secuencia.
        test_img_paths (list): Lista con las rutas de las imágenes de test.
        all_captions_dict (dict): Diccionario {nombre_imagen: [cap1, cap2...]} con las captions limpias.
        sample_size (int, optional): Si se define, solo evalúa esa cantidad de imágenes (para pruebas rápidas).
    """
    actual, predicted = [], []
    
    # Si queremos probar rápido, limitamos el número de imágenes
    eval_paths = test_img_paths[:sample_size] if sample_size else test_img_paths

    print(f"Calculando BLEU para {len(eval_paths)} imágenes...")

    for img_path in tqdm(eval_paths):
        # 1. Generar predicción del modelo
        # Nota: evaluate devuelve (result, attention_plot), solo queremos result
        pred_seq, _ = evaluate(img_path, encoder, decoder, tokenizer, max_len)
        predicted.append(pred_seq)
        
        # 2. Obtener referencias reales
        img_name = os.path.basename(img_path)
        
        # Obtenemos todas las captions reales para esa imagen
        # Las limpiamos y tokenizamos (split) para que coincidan con la predicción
        # Quitamos <start> y <end> si están en el diccionario para comparar solo el contenido
        raw_captions = all_captions_dict[img_name]
        references = []
        for c in raw_captions:
            # Asumiendo que c es "<start> un perro corre <end>"
            tokens = c.split()
            # Quitamos <start> (índice 0) y <end> (índice -1)
            # Ajusta esto según cómo hayas guardado tus captions limpias
            if tokens[0] == '<start>': tokens = tokens[1:]
            if tokens[-1] == '<end>': tokens = tokens[:-1]
            references.append(tokens)
            
        actual.append(references)

    # 3. Calcular BLEU Scores
    # BLEU-1: Coincidencia de palabras sueltas (unigramas)
    b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    # BLEU-4: Cuatrigramas (frases más largas)
    b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    print(f'\n--- Resultados BLEU ---')
    print(f'BLEU-1: {b1:.4f}')

    print(f'BLEU-4: {b4:.4f}')
    
    return b1, b4

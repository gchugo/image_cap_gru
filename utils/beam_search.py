import tensorflow as tf
import math
import numpy as np
import os
from tqdm.notebook import tqdm
from nltk.translate.bleu_score import corpus_bleu
from utils.eval import load_image_for_eval
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

# ==========================================
# 1. FUNCIÓN PRINCIPAL DE BEAM SEARCH
# ==========================================

def beam_search_evaluate(image_path, encoder, decoder, tokenizer, max_len, beam_width=3):
    """
    Genera un caption para una imagen usando el algoritmo Beam Search.
    """
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']

    # 1. Procesar imagen y obtener características
    # Usamos la función auxiliar importada de utils.eval
    img_tensor, _ = load_image_for_eval(image_path) 
    img_tensor = tf.expand_dims(img_tensor, 0)
    
    # Features shape: (1, 49, embedding_dim)
    features = encoder(img_tensor, training=False)

    # 2. Inicializar Decoder
    hidden = decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([start_token], 0)

    # Lista de candidatos: (secuencia, score, hidden_state)
    sequences = [([start_token], 0.0, hidden)]

    # 3. Bucle paso a paso
    for i in range(max_len):
        all_candidates = []
        
        for seq, score, hidden_state in sequences:
            # Si ya terminó, guardamos y pasamos
            if seq[-1] == end_token:
                all_candidates.append((seq, score, hidden_state))
                continue
            
            dec_input = tf.expand_dims([seq[-1]], 0)
            
            # Predicción
            predictions, new_hidden, _ = decoder(dec_input, features, hidden_state)
            predictions = tf.nn.softmax(predictions)
            
            # Top K
            top_k_probs, top_k_ids = tf.nn.top_k(predictions, k=beam_width)
            
            # Expandir
            for k in range(beam_width):
                word_id = top_k_ids[0][k].numpy()
                prob = top_k_probs[0][k].numpy()
                
                new_score = score + math.log(prob + 1e-20)
                new_seq = seq + [word_id]
                all_candidates.append((new_seq, new_score, new_hidden))
        
        # Selección (Poda)
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]

    # 4. Resultado final
    best_seq = sequences[0][0]
    
    # Decodificar IDs a palabras
    result_caption = [tokenizer.index_word[i] for i in best_seq if i not in [start_token, end_token]]
    
    return ' '.join(result_caption)


def calculate_python_metrics(encoder, decoder, tokenizer, max_len, test_paths, all_captions_dict, beam_width=5, sample_size=None):
    """
    Calcula BLEU, METEOR y ROUGE-L usando solo librerías Python (sin Java).
    """
    
    # Inicializar calculador ROUGE
    scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Listas para acumular
    actual_tokens = []    # Referencias tokenizadas (para BLEU)
    predicted_tokens = [] # Predicciones tokenizadas (para BLEU)
    
    meteor_scores = []    # Lista de scores METEOR individuales
    rouge_scores = []     # Lista de scores ROUGE individuales
    
    # Filtrar muestra
    eval_paths = test_paths[:sample_size] if sample_size else test_paths
        
    for img_path in tqdm(eval_paths):
        # 1. Generar Predicción (String)
        # beam_search_evaluate devuelve string ej: "un perro corre"
        pred_str = beam_search_evaluate(
            img_path, encoder, decoder, tokenizer, max_len, beam_width=beam_width
        )
        
        # Asegurarnos de que es string
        if isinstance(pred_str, list):
            pred_str = ' '.join(pred_str)
            
        pred_toks = pred_str.split() # Lista de palabras
        
        # 2. Obtener Referencias
        img_name = os.path.basename(img_path)
        raw_captions = all_captions_dict.get(img_name, [])
        
        ref_list_tokens = [] # Lista de listas de tokens (para METEOR/BLEU)
        ref_list_strs = []   # Lista de strings (para ROUGE)
        
        for c in raw_captions:
            # Limpieza simple
            c_clean = c.replace('<start>', '').replace('<end>', '').strip()
            ref_list_strs.append(c_clean)
            ref_list_tokens.append(c_clean.split())
            
        # --- ACUMULAR PARA BLEU ---
        actual_tokens.append(ref_list_tokens)
        predicted_tokens.append(pred_toks)
        
        # --- CALCULAR METEOR (Individual) ---
        # meteor_score toma (lista_de_referencias_tokenizadas, prediccion_tokenizada)
        # Nota: NLTK espera lista de strings tokens
        m_score = meteor_score(ref_list_tokens, pred_toks)
        meteor_scores.append(m_score)
        
        # --- CALCULAR ROUGE-L (Individual) ---
        # ROUGE compara la predicción contra CADA referencia y se queda con la mejor (o promedio)
        # Aquí tomamos la mejor coincidencia (max) de las 5 referencias
        best_rouge = 0
        for ref in ref_list_strs:
            scores = scorer_rouge.score(ref, pred_str)
            # Usamos fmeasure
            if scores['rougeL'].fmeasure > best_rouge:
                best_rouge = scores['rougeL'].fmeasure
        rouge_scores.append(best_rouge)

    
    # 1. BLEU
    b1 = corpus_bleu(actual_tokens, predicted_tokens, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-1:  {b1:.4f}")
    print(f"BLEU-2:  {b2:.4f}")
    print(f"BLEU-3:  {b3:.4f}")
    print(f"BLEU-4:  {b4:.4f}")
    
    # 2. METEOR (Promedio)
    avg_meteor = np.mean(meteor_scores)
    print(f"METEOR:  {avg_meteor:.4f}")
    
    # 3. ROUGE-L (Promedio)
    avg_rouge = np.mean(rouge_scores)
    print(f"ROUGE-L: {avg_rouge:.4f}")
    
    return {'bleu1': b1, 'bleu4': b4, 'meteor': avg_meteor, 'rouge': avg_rouge}


# ==========================================
# 2. FUNCIÓN PARA CALCULAR BLEU (MÉTRICAS)
# ==========================================

def calculate_bleu_beam(encoder, decoder, tokenizer, max_len, test_paths, all_captions_dict, beam_width=5, sample_size=None):
    """
    Calcula métricas BLEU sobre el set de test usando Beam Search.
    """
    actual, predicted = [], []
    
    # Si definimos sample_size, cortamos la lista para pruebas rápidas
    eval_paths = test_paths[:sample_size] if sample_size else test_paths

    print(f"Calculando BLEU (Beam={beam_width}) para {len(eval_paths)} imágenes...")

    for img_path in tqdm(eval_paths):
        # 1. Generar predicción con BEAM SEARCH
        # Llamamos directamente a la función definida arriba (sin import extra)
        caption_str = beam_search_evaluate(
            img_path, encoder, decoder, tokenizer, max_len, beam_width=beam_width
        )
        
        # corpus_bleu necesita lista de tokens
        predicted.append(caption_str.split())
        
        # 2. Obtener referencias reales
        img_name = os.path.basename(img_path)
        
        # Manejo de errores si la imagen no está en el dict (poco probable pero posible)
        if img_name not in all_captions_dict:
            print(f"Warning: {img_name} no encontrada en captions dict.")
            continue

        raw_captions = all_captions_dict[img_name]
        references = []
        for c in raw_captions:
            tokens = c.split()
            if tokens[0] == '<start>': tokens = tokens[1:]
            if tokens[-1] == '<end>': tokens = tokens[:-1]
            references.append(tokens)
            
        actual.append(references)

    # 3. Calcular Scores
    print("Calculando métricas finales...")
    b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    print(f'\n--- Resultados BLEU (Beam Width {beam_width}) ---')
    print(f'BLEU-1: {b1:.4f}')
    print(f'BLEU-2: {b2:.4f}')
    print(f'BLEU-3: {b3:.4f}')
    print(f'BLEU-4: {b4:.4f}')
    
    return b1, b2, b3, b4

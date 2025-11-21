import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_tokenizer(all_captions, top_k=10000):
    """Crea y ajusta el tokenizador."""
    tokenizer = Tokenizer(num_words=top_k, oov_token="<unk>", filters='', lower=False)
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

def save_tokenizer(tokenizer, path='tokenizer.pkl'):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(path='tokenizer.pkl'):
    with open(path, 'rb') as handle:
        return pickle.load(handle)
    
def calc_max_length(tensor):
    return max(len(t.split()) for t in tensor)

def text_to_padded_sequences(captions_text_list, tokenizer, max_length):
    # 1. Texto a números
    seqs = tokenizer.texts_to_sequences(captions_text_list)
    
    # 2. Padding (rellenar con ceros al final hasta llegar a max_length)
    # 'post' significa que los ceros van al final: [12, 45, 0, 0]
    seqs_padded = pad_sequences(seqs, maxlen=max_length, padding='post')
    
    return seqs_padded
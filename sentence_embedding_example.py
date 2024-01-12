from sentence_transformers import SentenceTransformer

# Caricare un modello SBERT preaddestrato
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Frase di esempio
sentence = "Questo è un esempio di frase"

# Ottenere l'embedding della frase
sentence_embedding = model.encode(sentence, convert_to_numpy=True).tolist()
print(f"Sentence: {sentence}")
print(f"Embedding: {sentence_embedding}")
print(f"Embedding Len: {len(sentence_embedding)}")
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def embed_text(text):
    embeddings = model.encode([text])
    return embeddings[0]

def calculate_cosine_similarity(text1, text2):
    embedding1 = embed_text(text1)
    embedding2 = embed_text(text2)
    cos_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return cos_sim

if __name__ == "__main__":
    text1 = "This is a sample sentence."
    text2 = "This is another example sentence."

    similarity = calculate_cosine_similarity(text1, text2)
    print(f"Cosine Similarity: {similarity}")
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# สร้าง index FAISS
dimension = 384
index = faiss.IndexFlatL2(dimension)

# mapping (id → song/segment)
metadata = []

def add_segments_to_index(song_id, segments):
    vectors = embedder.encode(segments)
    index.add(np.array(vectors, dtype="float32"))
    for i in range(len(segments)):
        metadata.append((song_id, i))

def search_query(query, top_k=5):
    vec = embedder.encode([query])
    D, I = index.search(np.array(vec, dtype="float32"), top_k)
    results = [metadata[i] for i in I[0]]
    return results
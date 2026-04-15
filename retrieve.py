from Parser import load_documents
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


DOCUMENTS = []
GLOSSARY = {}
INITIALIZED = False

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDINGS = None
TFIDF = None
TFIDF_MATRIX = None

#Init
def init():
    global DOCUMENTS, GLOSSARY, INITIALIZED

    if not INITIALIZED:
        DOCUMENTS, GLOSSARY = load_documents()
        INITIALIZED = True


# Query type detection
def is_variable_query(query):
    return "_" in query


# Glossary lookup
def glossary_lookup(query):
    query_lower = query.lower()

    for key, definition in GLOSSARY.items():
        if key.lower() in query_lower:
            return [{
                "text": definition,
                "source": "glossary",
                "score": 1.0
            }]
    return None


# Build TF-IDF
def build_tfidf():
    global TFIDF, TFIDF_MATRIX

    texts = [d["text"] for d in DOCUMENTS]
    TFIDF = TfidfVectorizer(stop_words="english")
    TFIDF_MATRIX = TFIDF.fit_transform(texts)


# Build embeddings
def build_embeddings():
    global EMBEDDINGS

    texts = [d["text"] for d in DOCUMENTS]
    EMBEDDINGS = MODEL.encode(texts, convert_to_tensor=True)


# Retrieval function
def retrieve(query: str) -> list[dict]:
    init()

    #Narrow queries
    if is_variable_query(query):
        result = glossary_lookup(query)
        if result:
            return result

    global TFIDF, EMBEDDINGS

    # Building index
    if TFIDF is None:
        build_tfidf()

    if EMBEDDINGS is None:
        build_embeddings()

    #Embedding similarity
    query_emb = MODEL.encode(query, convert_to_tensor=True)
    emb_scores = util.cos_sim(query_emb, EMBEDDINGS)[0].cpu().numpy()

    #TF-IDF similarity
    query_vec = TFIDF.transform([query])
    tfidf_scores = (TFIDF_MATRIX @ query_vec.T).toarray().flatten()

    #Hybrid scoring
    final_scores = 0.6 * emb_scores + 0.4 * tfidf_scores

    #Top 5 results
    top_idx = np.argsort(final_scores)[::-1][:5]

    results = []
    for i in top_idx:
        results.append({
            "text": DOCUMENTS[i]["text"],
            "source": DOCUMENTS[i]["source"],
            "score": float(final_scores[i])
        })

    return results
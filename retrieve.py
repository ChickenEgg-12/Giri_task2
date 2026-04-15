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


# Init
def init():
    global DOCUMENTS, GLOSSARY, INITIALIZED

    if not INITIALIZED:
        DOCUMENTS, GLOSSARY = load_documents()
        INITIALIZED = True

# Normalize query (synonyms)
def normalize_query(query):
    query = query.lower()

    synonyms = {
        "share of wallet": "share_of_wallet",
        "brand awareness": "frm_brand_awareness",
        "trialists": "probable_trialists"
    }

    for k, v in synonyms.items():
        if k in query:
            query += " " + v

    return query


# Detect query type
def is_variable_query(query):
    if "_" in query:
        return True

    keywords = ["what does", "define", "what is", "mean"]
    if any(k in query for k in keywords):
        return True

    return False


# Improved glossary lookup
def glossary_lookup(query):
    query_lower = query.lower()

    best_match = None
    best_score = 0

    for key, definition in GLOSSARY.items():

        #Exact match
        if key in query_lower:
            return {
                "text": f"{key.replace('_', ' ')}: {definition}",
                "source": "glossary",
                "score": 1.0
            }

        #Token overlap
        key_tokens = key.replace("_", " ").split()
        query_tokens = query_lower.split()

        overlap = len(set(key_tokens) & set(query_tokens))

        if overlap > best_score:
            best_score = overlap
            best_match = (key, definition)

    if best_match and best_score >= 2:
        return {
            "text": best_match[1],
            "source": "glossary",
            "score": 0.95
        }

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


#Retrieve function
def retrieve(query: str) -> list[dict]:
    init()

    global TFIDF, EMBEDDINGS

    #Normalize query
    query = normalize_query(query)

    #Build indexes
    if TFIDF is None:
        build_tfidf()

    if EMBEDDINGS is None:
        build_embeddings()

    #Embedding scores
    query_emb = MODEL.encode(query, convert_to_tensor=True)
    emb_scores = util.cos_sim(query_emb, EMBEDDINGS)[0].cpu().numpy()

    #TF-IDF scores
    query_vec = TFIDF.transform([query])
    tfidf_scores = (TFIDF_MATRIX @ query_vec.T).toarray().flatten()

    #Hybrid scoring
    final_scores = 0.6 * emb_scores + 0.4 * tfidf_scores

    #Glossary result
    glossary_result = None
    if is_variable_query(query):
        glossary_result = glossary_lookup(query)

    #Top results
    top_idx = np.argsort(final_scores)[::-1][:5]

    results = []
    for i in top_idx:
        results.append({
            "text": DOCUMENTS[i]["text"],
            "source": DOCUMENTS[i]["source"],
            "score": float(final_scores[i])
        })

    #Add glossary at top
    if glossary_result:
        results.insert(0, glossary_result)
        results = results[:5]

    return results
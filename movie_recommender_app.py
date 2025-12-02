# Usage: streamlit run movie_recommender_app.py

import streamlit as st
import pandas as pd
import ast
import numpy as np
import os
import hashlib
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize

# ---------- CONFIG ----------
MOVIES_CSV = "tmdb_5000_movies.csv"
CREDITS_CSV = "tmdb_5000_credits.csv"
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # fast, high-quality for many tasks
EMBEDDINGS_DIR = "embeddings_cache"
EMBEDDINGS_DTYPE = np.float32
# threshold for switching to ANN; for small datasets brute-force dot product is fine
ANN_THRESHOLD = 5000
# ----------------------------

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ---------- Utilities ----------

def file_md5(path):
    """Compute md5 of a file for change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_names_from_literal(text):
    if not isinstance(text, str):
        return []
    try:
        data = ast.literal_eval(text)
        names = [d.get("name", "") for d in data if isinstance(d, dict)]
        return names
    except Exception:
        return []


def extract_cast_names(text, top_k=5):
    if not isinstance(text, str):
        return []
    try:
        data = ast.literal_eval(text)
        names = [d.get("name", "") for d in data if isinstance(d, dict)]
        return names[:top_k]
    except Exception:
        return []


def extract_directors(crew_text):
    if not isinstance(crew_text, str):
        return []
    try:
        data = ast.literal_eval(crew_text)
        directors = [d.get("name") for d in data if isinstance(d, dict) and d.get("job") == "Director"]
        return directors
    except Exception:
        return []


def build_document(row):
    parts = []
    title = str(row.get("title", ""))
    parts.append(title)
    if "release_date" in row and isinstance(row["release_date"], str) and len(row["release_date"]) >= 4:
        parts.append(row["release_date"][:4])
    parts.append(row.get("overview", ""))
    parts.extend([g.replace(" ", "_") for g in row.get("genres_list", []) if g])
    parts.extend([k.replace(" ", "_") for k in row.get("keywords_list", []) if k])
    parts.extend([c.replace(" ", "_") for c in row.get("cast_list", []) if c])
    parts.extend([d.replace(" ", "_") for d in row.get("directors", []) if d])
    return " ".join([p for p in parts if p])


@st.cache_data
def load_and_merge(movies_path=MOVIES_CSV, credits_path=CREDITS_CSV):
    """Load csvs, normalize column names, find title column robustly,
       merge, parse lists, and build a textual document per movie."""
    # --- load CSVs ---
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # --- normalize column names: strip whitespace & lower for safe checks ---
    movies.columns = movies.columns.map(lambda c: c.strip() if isinstance(c, str) else c)
    credits.columns = credits.columns.map(lambda c: c.strip() if isinstance(c, str) else c)

    # --- detect title column in movies: try several common options ---
    title_candidates = ['title', 'original_title', 'name']
    found_title_col = None
    for cand in title_candidates:
        if cand in movies.columns:
            found_title_col = cand
            break
    # if not found, try any column that contains 'title' substring (case-insensitive)
    if found_title_col is None:
        for c in movies.columns:
            if isinstance(c, str) and 'title' in c.lower():
                found_title_col = c
                break

    if found_title_col is None:
        # fallback: create a synthetic title column using index to avoid missing column errors
        movies['title'] = movies.index.astype(str)
    else:
        # unify into 'title' column and ensure strings
        movies['title'] = movies[found_title_col].astype(str)

    # also normalize credits title column if exists
    if 'title' not in credits.columns:
        # try to find candidate in credits too
        for cand in title_candidates:
            if cand in credits.columns:
                credits['title'] = credits[cand].astype(str)
                break
    else:
        credits['title'] = credits['title'].astype(str)

    # --- Merge robustly: prefer id/movie_id, otherwise merge on cleaned title ---
    # strip whitespace and lower-case titles for more reliable merge
    movies['__join_title'] = movies['title'].fillna('').astype(str).str.strip().str.lower()
    credits['__join_title'] = credits['title'].fillna('').astype(str).str.strip().str.lower()

    if 'id' in movies.columns and 'movie_id' in credits.columns:
        merged = pd.merge(movies, credits, left_on='id', right_on='movie_id', how='left', suffixes=('', '_cred'))
    else:
        merged = pd.merge(movies, credits, left_on='__join_title', right_on='__join_title', how='left', suffixes=('', '_cred'))

    # ensure we have a clean 'title' column in merged DF (take movies.title primarily)
    merged['title'] = merged.get('title', merged.get('title_cred', merged.get(found_title_col, merged.index.astype(str)))).astype(str)

    # --- parse fields safely ---
    merged['genres_list'] = merged['genres'].apply(extract_names_from_literal) if 'genres' in merged.columns else [[] for _ in range(len(merged))]
    merged['keywords_list'] = merged['keywords'].apply(extract_names_from_literal) if 'keywords' in merged.columns else [[] for _ in range(len(merged))]
    merged['cast_list'] = merged['cast'].apply(lambda x: extract_cast_names(x, top_k=5)) if 'cast' in merged.columns else [[] for _ in range(len(merged))]
    merged['directors'] = merged['crew'].apply(extract_directors) if 'crew' in merged.columns else [[] for _ in range(len(merged))]
    merged['overview'] = merged['overview'].fillna('') if 'overview' in merged.columns else [''] * len(merged)

    # --- build document and ensure title column has no empty strings ---
    def build_doc(row):
        parts = []
        title = str(row.get('title', '')).strip()
        if title:
            parts.append(title)
        if 'release_date' in row and isinstance(row['release_date'], str) and len(row['release_date']) >= 4:
            parts.append(row['release_date'][:4])
        parts.append(row.get('overview', '') or '')
        parts.extend([g.replace(' ', '_') for g in row.get('genres_list', []) if g])
        parts.extend([k.replace(' ', '_') for k in row.get('keywords_list', []) if k])
        parts.extend([c.replace(' ', '_') for c in row.get('cast_list', []) if c])
        parts.extend([d.replace(' ', '_') for d in row.get('directors', []) if d])
        return " ".join([p for p in parts if p])

    merged['document'] = merged.apply(build_doc, axis=1)

    # ensure 'title' field is filled: if empty, try to extract from document (first token(s))
    def ensure_title(row):
        t = str(row.get('title', '')).strip()
        if t:
            return t
        doc = str(row.get('document', '')).strip()
        if doc:
            # take first up-to-4 tokens as fallback title (replace underscores)
            tokens = doc.split()
            fallback = " ".join(tokens[:4]).replace('_', ' ')
            return fallback
        return ''

    merged['title'] = merged.apply(ensure_title, axis=1)

    merged = merged[merged['document'].str.strip().astype(bool)].reset_index(drop=True)
    return merged


@st.cache_resource
def load_model(model_name=DEFAULT_MODEL):
    """Load SBERT model and cache resource across reruns."""
    return SentenceTransformer(model_name)


def embeddings_cache_path(data_hash, model_name):
    safe_model = model_name.replace("/", "_")
    return os.path.join(EMBEDDINGS_DIR, f"emb_{safe_model}_{data_hash}.npy")


@st.cache_data
def encode_movies(documents, model_name=DEFAULT_MODEL, force_recompute=False, data_hash=None):
    """Encode documents into normalized float32 embeddings. If cache file exists, load it."""
    cache_path = embeddings_cache_path(data_hash or "nodata", model_name)
    if os.path.exists(cache_path) and not force_recompute:
        emb = np.load(cache_path)
        return emb

    model = load_model(model_name)
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    # normalize and cast to float32
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    embeddings = embeddings.astype(EMBEDDINGS_DTYPE)
    np.save(cache_path, embeddings)
    return embeddings


def try_import_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None


def build_faiss_index(movie_embeddings):
    faiss = try_import_faiss()
    if faiss is None:
        return None
    d = movie_embeddings.shape[1]
    # use inner product on normalized vectors to approximate cosine similarity
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    index.add(movie_embeddings)
    return index


def ann_search(index, q_emb, top_k=10):
    # faiss expects (n, d) float32
    D, I = index.search(q_emb.astype(np.float32), top_k)
    # if using IndexHNSWFlat and normalized vectors, D is inner product (higher better)
    return I[0], D[0]


def brute_force_search(movie_embeddings, q_emb, top_k=10):
    sims = np.dot(movie_embeddings, q_emb.T).squeeze(1)
    top_idx = np.argsort(-sims)[:top_k]
    return top_idx, sims[top_idx]


def re_rank_with_crossencoder(query, docs_texts, indices, cross_model_name):
    try:
        cross = CrossEncoder(cross_model_name)
    except Exception:
        return indices  # fallback if model cannot be loaded

    pairs = [(query, docs_texts[i]) for i in indices]
    scores = cross.predict(pairs)
    order = np.argsort(-scores)
    return [indices[i] for i in order]


# ---------- Streamlit UI ----------

def get_top_k_recommendations(query, movie_embeddings, movies_df, model, top_k=10, faiss_index=None, use_crossencoder=False, crossencoder_name=None, min_score=0.20):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, ord=2, axis=1, keepdims=True)
    q_emb = q_emb.astype(EMBEDDINGS_DTYPE)

    if faiss_index is not None:
        idxs, scores = ann_search(faiss_index, q_emb, top_k=top_k)
    else:
        idxs, scores = brute_force_search(movie_embeddings, q_emb, top_k=top_k)

    if use_crossencoder and crossencoder_name and len(idxs) > 0:
        # rerank top results with cross encoder
        texts = movies_df['document'].tolist()
        idxs = re_rank_with_crossencoder(query, texts, idxs.tolist(), crossencoder_name)
        scores = [float(0.0)] * len(idxs)  # cross scores not returned here, keep placeholder

    results = []
    for i, idx in enumerate(np.atleast_1d(idxs)):
        score = float(scores[i]) if isinstance(scores, (list, np.ndarray)) else float(scores)
        if score < min_score and i >= 3:
            # if score low and not top result, stop adding more
            break
        row = movies_df.iloc[int(idx)]
        results.append({
            "title": row.get('title', ''),
            "release_date": row.get('release_date', ''),
            "overview": row.get('overview', ''),
            "genres": ", ".join(row.get('genres_list', [])),
            "cast": ", ".join(row.get('cast_list', [])),
            "score": score
        })
    return results


def main():
    st.set_page_config(page_title="Chatbot Movie Recommender", layout="wide")
    st.title("Chatbot Movie Recommender (SBERT + Cosine Sim + Optional ANN)")
    st.markdown("Type a short description like: 'serial thriller about secret agent' and get movie recommendations.")

    with st.spinner("Loading dataset..."):
        try:
            movies_df = load_and_merge(MOVIES_CSV, CREDITS_CSV)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    st.sidebar.header("Settings")
    model_name = st.sidebar.text_input("SBERT model name", value=DEFAULT_MODEL)
    top_k = st.sidebar.slider("Number of results to show", min_value=3, max_value=20, value=10)
    use_ann = st.sidebar.checkbox("Use ANN (FAISS) if available", value=True)
    use_cross = st.sidebar.checkbox("Use cross-encoder re-ranking (slower)", value=False)
    cross_model_name = st.sidebar.text_input("Cross-encoder model name", value="cross-encoder/ms-marco-MiniLM-L-6-v2") if use_cross else None
    min_score = st.sidebar.slider("Minimum similarity score (0-1)", min_value=0.0, max_value=1.0, value=0.18, step=0.01)

    # compute data hash from CSVs to version embeddings
    try:
        movies_hash = file_md5(MOVIES_CSV)
        credits_hash = file_md5(CREDITS_CSV)
    except Exception:
        movies_hash = "nofilehash"
        credits_hash = "nofilehash"
    data_hash = hashlib.md5((movies_hash + credits_hash).encode()).hexdigest()[:8]

    # load or compute embeddings
    emb_cache_path = embeddings_cache_path(data_hash, model_name)
    if 'movie_embeddings' not in st.session_state or st.session_state.get('emb_cache') != emb_cache_path:
        with st.spinner("Loading SBERT model and encoding movies (may take a while)..."):
            try:
                movie_docs = movies_df['document'].tolist()
                movie_embeddings = encode_movies(movie_docs, model_name=model_name, data_hash=data_hash)
                st.session_state['movie_embeddings'] = movie_embeddings
                st.session_state['emb_cache'] = emb_cache_path
                st.session_state['sbert_model'] = load_model(model_name)
            except Exception as e:
                st.error(f"Failed to encode or load model: {e}")
                st.stop()
    else:
        movie_embeddings = st.session_state['movie_embeddings']

    # build or load ANN index if requested and dataset large
    faiss_index = None
    faiss = None
    if use_ann and movie_embeddings.shape[0] >= ANN_THRESHOLD:
        faiss = try_import_faiss()
        if faiss is not None:
            if 'faiss_index' not in st.session_state or st.session_state.get('faiss_idx_size') != movie_embeddings.shape[0]:
                with st.spinner("Building FAISS HNSW index for fast search..."):
                    try:
                        faiss_index = build_faiss_index(movie_embeddings)
                        st.session_state['faiss_index'] = faiss_index
                        st.session_state['faiss_idx_size'] = movie_embeddings.shape[0]
                    except Exception:
                        st.session_state['faiss_index'] = None
                        st.session_state['faiss_idx_size'] = 0
            else:
                faiss_index = st.session_state.get('faiss_index')
        else:
            st.sidebar.warning("faiss not installed. Falling back to exact search.")

    # chat input
    user_input = st.text_input("Ask me for movie recommendations:", placeholder="e.g. superhero film where the hero has supernatural power")
    if st.button("Recommend") and user_input:
        with st.spinner("Searching for best matches..."):
            model = st.session_state.get('sbert_model')
            results = get_top_k_recommendations(user_input, movie_embeddings, movies_df, model,
                                                top_k=top_k, faiss_index=faiss_index if faiss is not None else None,
                                                use_crossencoder=use_cross, crossencoder_name=cross_model_name,
                                                min_score=min_score)

        if not results:
            st.info("No good matches found. Try a different query or make your query more specific.")
        else:
            st.subheader(f"Top {len(results)} recommendations for: \"{user_input}\"")
            for i, r in enumerate(results, 1):
                st.markdown(f"**{i}. {r['title']}**  ")
                st.write(f"Score: {r['score']:.4f}  ")
                st.write(f"Release date: {r.get('release_date','N/A')}  ")
                st.write(r.get('overview',''))
                st.write(f"Genres: {r.get('genres','N/A')}")
                st.write(f"Cast: {r.get('cast','N/A')}")
                st.markdown("---")


if __name__ == "__main__":
    main()

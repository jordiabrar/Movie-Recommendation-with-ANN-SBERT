# movie_recommender_app.py
# Usage: streamlit run movie_recommender_app.py

import streamlit as st
import pandas as pd
import ast
import numpy as np
import os
import hashlib
from sentence_transformers import SentenceTransformer

# Keras / TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------- CONFIG ----------
MOVIES_CSV = "tmdb_5000_movies.csv"
CREDITS_CSV = "tmdb_5000_credits.csv"
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # SBERT model for embeddings
EMBEDDINGS_DIR = "embeddings_cache"
EMBEDDINGS_DTYPE = np.float32

ANN_MODEL_DIR = "ann_models"
# ----------------------------

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(ANN_MODEL_DIR, exist_ok=True)

# ---------- Utilities ----------

def file_md5(path):
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
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    movies.columns = movies.columns.map(lambda c: c.strip() if isinstance(c, str) else c)
    credits.columns = credits.columns.map(lambda c: c.strip() if isinstance(c, str) else c)

    title_candidates = ['title', 'original_title', 'name']
    found_title_col = None
    for cand in title_candidates:
        if cand in movies.columns:
            found_title_col = cand
            break
    if found_title_col is None:
        for c in movies.columns:
            if isinstance(c, str) and 'title' in c.lower():
                found_title_col = c
                break

    if found_title_col is None:
        movies['title'] = movies.index.astype(str)
    else:
        movies['title'] = movies[found_title_col].astype(str)

    if 'title' not in credits.columns:
        for cand in title_candidates:
            if cand in credits.columns:
                credits['title'] = credits[cand].astype(str)
                break
    else:
        credits['title'] = credits['title'].astype(str)

    movies['__join_title'] = movies['title'].fillna('').astype(str).str.strip().str.lower()
    credits['__join_title'] = credits['title'].fillna('').astype(str).str.strip().str.lower()

    if 'id' in movies.columns and 'movie_id' in credits.columns:
        merged = pd.merge(movies, credits, left_on='id', right_on='movie_id', how='left', suffixes=('', '_cred'))
    else:
        merged = pd.merge(movies, credits, left_on='__join_title', right_on='__join_title', how='left', suffixes=('', '_cred'))

    merged['title'] = merged.get('title', merged.get('title_cred', merged.get(found_title_col, merged.index.astype(str)))).astype(str)

    merged['genres_list'] = merged['genres'].apply(extract_names_from_literal) if 'genres' in merged.columns else [[] for _ in range(len(merged))]
    merged['keywords_list'] = merged['keywords'].apply(extract_names_from_literal) if 'keywords' in merged.columns else [[] for _ in range(len(merged))]
    merged['cast_list'] = merged['cast'].apply(lambda x: extract_cast_names(x, top_k=5)) if 'cast' in merged.columns else [[] for _ in range(len(merged))]
    merged['directors'] = merged['crew'].apply(extract_directors) if 'crew' in merged.columns else [[] for _ in range(len(merged))]
    merged['overview'] = merged['overview'].fillna('') if 'overview' in merged.columns else [''] * len(merged)

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

    def ensure_title(row):
        t = str(row.get('title', '')).strip()
        if t:
            return t
        doc = str(row.get('document', '')).strip()
        if doc:
            tokens = doc.split()
            fallback = " ".join(tokens[:4]).replace('_', ' ')
            return fallback
        return ''

    merged['title'] = merged.apply(ensure_title, axis=1)
    merged = merged[merged['document'].str.strip().astype(bool)].reset_index(drop=True)
    return merged

@st.cache_resource
def load_sbert(model_name=DEFAULT_MODEL):
    return SentenceTransformer(model_name)

def embeddings_cache_path(data_hash, model_name, view_name):
    safe_model = model_name.replace("/", "_")
    return os.path.join(EMBEDDINGS_DIR, f"emb_{safe_model}_{view_name}_{data_hash}.npy")

@st.cache_data
def encode_documents(documents, model_name=DEFAULT_MODEL, force_recompute=False, data_hash=None, view_name="a"):
    # validate input
    if documents is None:
        raise ValueError("No documents provided to encode.")
    if not hasattr(documents, "__len__"):
        raise ValueError("Documents must be a list-like of strings.")
    # convert to list of str, ensure no None
    docs = ["" if d is None else str(d) for d in documents]
    # create cache path
    cache_path = embeddings_cache_path(data_hash or "nodata", model_name, view_name)
    if os.path.exists(cache_path) and not force_recompute:
        emb = np.load(cache_path)
        return emb

    model = load_sbert(model_name)
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    embeddings = embeddings.astype(EMBEDDINGS_DTYPE)
    np.save(cache_path, embeddings)
    return embeddings

def projector_model_path(data_hash):
    return os.path.join(ANN_MODEL_DIR, f"proj_model_{data_hash}.h5")

def projections_path(data_hash):
    return os.path.join(ANN_MODEL_DIR, f"movie_projs_{data_hash}.npy")

def build_projector(input_dim, proj_dim=256, hidden_sizes=[512, 256], dropout=0.2):
    inp = layers.Input(shape=(input_dim,), name="sbert_input")
    x = inp
    for i, h in enumerate(hidden_sizes):
        x = layers.Dense(h, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(proj_dim, name="proj")(x)
    x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2norm")(x)
    model = Model(inputs=inp, outputs=x, name="ann_projector")
    return model

@tf.function
def nt_xent_loss(za, zb, temperature=0.07):
    batch_size = tf.shape(za)[0]
    logits = tf.matmul(za, zb, transpose_b=True) / temperature
    labels = tf.range(batch_size)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    loss_a = loss_fn(labels, logits)
    loss_b = loss_fn(labels, tf.transpose(logits))
    loss = 0.5 * (tf.reduce_mean(loss_a) + tf.reduce_mean(loss_b))
    return loss

def train_projector(projector, emb_a, emb_b, epochs=10, batch_size=256, lr=1e-3, temperature=0.07,
                    val_split=0.05, patience=5, save_path=None):
    n = emb_a.shape[0]
    assert emb_a.shape == emb_b.shape
    if val_split and val_split > 0 and val_split < 1.0:
        idx = np.arange(n)
        np.random.shuffle(idx)
        cut = int(n * (1 - val_split))
        train_idx, val_idx = idx[:cut], idx[cut:]
        train_a, train_b = emb_a[train_idx], emb_b[train_idx]
        val_a, val_b = emb_a[val_idx], emb_b[val_idx]
    else:
        train_a, train_b = emb_a, emb_b
        val_a = val_b = None

    dataset = tf.data.Dataset.from_tensor_slices((train_a, train_b))
    dataset = dataset.shuffle(buffer_size=max(1, train_a.shape[0])).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if val_a is not None:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_a, val_b)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_dataset = None

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    best_val = np.inf
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        it = 0
        for batch_a, batch_b in dataset:
            with tf.GradientTape() as tape:
                za = projector(batch_a, training=True)
                zb = projector(batch_b, training=True)
                loss = nt_xent_loss(za, zb, temperature)
            grads = tape.gradient(loss, projector.trainable_variables)
            optimizer.apply_gradients(zip(grads, projector.trainable_variables))
            total_loss += float(loss)
            it += 1
        avg_train_loss = total_loss / max(1, it)
        history["train_loss"].append(avg_train_loss)

        if val_dataset is not None:
            total_val = 0.0
            itv = 0
            for vb_a, vb_b in val_dataset:
                za_v = projector(vb_a, training=False)
                zb_v = projector(vb_b, training=False)
                vloss = nt_xent_loss(za_v, zb_v, temperature)
                total_val += float(vloss)
                itv += 1
            avg_val_loss = total_val / max(1, itv)
            history["val_loss"].append(avg_val_loss)
        else:
            avg_val_loss = avg_train_loss
            history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val - 1e-6:
            best_val = avg_val_loss
            wait = 0
            if save_path:
                try:
                    projector.save(save_path, include_optimizer=False)
                except Exception:
                    projector.save_weights(save_path + ".weights.h5")
        else:
            wait += 1
            if wait >= patience:
                break

    return projector, history

def compute_movie_projections(projector, movie_embeddings, batch_size=1024):
    n = movie_embeddings.shape[0]
    parts = []
    for i in range(0, n, batch_size):
        batch = movie_embeddings[i:i+batch_size].astype(np.float32)
        p = projector(batch, training=False).numpy()
        parts.append(p)
    allp = np.vstack(parts)
    norms = np.linalg.norm(allp, axis=1, keepdims=True)
    norms[norms == 0] = 1
    allp = allp / norms
    return allp.astype(np.float32)

# ---------- Streamlit UI & Logic ----------

def make_view_a(row):
    return row.get("document", "")

def make_view_b(row):
    parts = []
    title = row.get("title", "")
    if title:
        parts.append(title)
    overview = (row.get("overview") or "").strip()
    if overview:
        words = overview.split()
        parts.append(" ".join(words[:30]))
    genres = row.get("genres_list", [])
    if genres:
        parts.append(genres[0])
    keywords = row.get("keywords_list", [])
    if keywords:
        parts.append(keywords[0])
    cast = row.get("cast_list", [])
    if cast:
        parts.append(cast[0])
    return " ".join([p for p in parts if p])

def ensure_views_exist(df):
    # create missing view columns robustly
    if 'view_a' not in df.columns:
        df['view_a'] = df.apply(make_view_a, axis=1)
    if 'view_b' not in df.columns:
        df['view_b'] = df.apply(make_view_b, axis=1)
    # cast to str and fillna
    df['view_a'] = df['view_a'].fillna('').astype(str)
    df['view_b'] = df['view_b'].fillna('').astype(str)
    return df

def main():
    st.set_page_config(page_title="ANN Movie Recommender (SBERT -> Projector ANN)", layout="wide")
    st.title("ANN Movie Recommender - SBERT for embedding, ANN as projector")
    st.markdown("Pipeline: two views -> SBERT encode -> ANN projector trained via contrastive loss -> retrieval by cosine similarity.")

    with st.spinner("Loading dataset..."):
        try:
            movies_df = load_and_merge(MOVIES_CSV, CREDITS_CSV)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    st.sidebar.header("Settings")
    model_name = st.sidebar.text_input("SBERT model name", value=DEFAULT_MODEL)
    top_k = st.sidebar.slider("Top K results", min_value=3, max_value=50, value=10)
    proj_dim = st.sidebar.number_input("Projection dim", min_value=32, max_value=2048, value=256)
    hidden1 = st.sidebar.number_input("Hidden size 1", min_value=32, max_value=4096, value=512)
    hidden2 = st.sidebar.number_input("Hidden size 2", min_value=0, max_value=4096, value=256)
    dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.8, value=0.2)
    epochs = st.sidebar.number_input("Training epochs", min_value=1, max_value=500, value=20)
    batch_size = st.sidebar.number_input("Training batch size", min_value=8, max_value=4096, value=256)
    lr = st.sidebar.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")
    temperature = st.sidebar.number_input("NT-Xent temperature", min_value=0.01, max_value=1.0, value=0.07, format="%.3f")
    force_reencode = st.sidebar.checkbox("Force re-encode documents", value=False)
    force_reproj = st.sidebar.checkbox("Force recompute projections", value=False)

    try:
        movies_hash = file_md5(MOVIES_CSV)
        credits_hash = file_md5(CREDITS_CSV)
    except Exception:
        movies_hash = "nofilehash"
        credits_hash = "nofilehash"
    data_hash = hashlib.md5((movies_hash + credits_hash).encode()).hexdigest()[:8]

    st.header("Prepare views & encode")
    st.write("View A = full document. View B = title + short overview + first genre/keyword/cast.")

    # always ensure views exist for current movies_df
    movies_df = ensure_views_exist(movies_df)
    st.session_state['views_ready'] = True

    if 'sbert_model' not in st.session_state or st.session_state.get('sbert_name') != model_name:
        with st.spinner("Loading SBERT model..."):
            try:
                sbert = load_sbert(model_name)
                st.session_state['sbert_model'] = sbert
                st.session_state['sbert_name'] = model_name
            except Exception as e:
                st.error(f"Failed to load SBERT: {e}")
                st.stop()
    else:
        sbert = st.session_state['sbert_model']

    emb_a_path = embeddings_cache_path(data_hash, model_name, "viewA")
    emb_b_path = embeddings_cache_path(data_hash, model_name, "viewB")

    if st.button("Encode views with SBERT"):
        with st.spinner("Encoding view A and view B..."):
            try:
                # make absolutely sure columns exist and are strings
                movies_df = ensure_views_exist(movies_df)
                docs_a = movies_df['view_a'].astype(str).fillna('').tolist()
                docs_b = movies_df['view_b'].astype(str).fillna('').tolist()
                if len(docs_a) == 0 or len(docs_b) == 0:
                    st.error("No documents to encode. Check dataset and views.")
                else:
                    emb_a = encode_documents(docs_a, model_name=model_name, data_hash=data_hash, force_recompute=force_reencode, view_name="viewA")
                    emb_b = encode_documents(docs_b, model_name=model_name, data_hash=data_hash, force_recompute=force_reencode, view_name="viewB")
                    st.session_state['emb_a'] = emb_a
                    st.session_state['emb_b'] = emb_b
                    st.success(f"Encoded: saved to {emb_a_path} and {emb_b_path}")
            except Exception as e:
                # more informative error report
                st.error(f"Encoding failed: {type(e).__name__}: {e}")
                # also print a helpful hint
                st.write("Hint: make sure TMDB CSV files are present and contain expected columns like 'overview', 'genres'.")
                st.stop()

    # Try to load embeddings from session or disk
    if 'emb_a' not in st.session_state:
        if os.path.exists(emb_a_path) and os.path.exists(emb_b_path) and not force_reencode:
            try:
                st.session_state['emb_a'] = np.load(emb_a_path)
                st.session_state['emb_b'] = np.load(emb_b_path)
                st.success("Loaded cached embeddings from disk.")
            except Exception:
                st.info("Embeddings cache not loadable; click 'Encode views with SBERT'.")
        else:
            st.info("No embeddings cached. Click 'Encode views with SBERT' to create them.")

    if 'emb_a' in st.session_state:
        emb_a = st.session_state['emb_a']
        emb_b = st.session_state['emb_b']
        st.sidebar.markdown(f"**Dataset:** {emb_a.shape[0]} items  •  SBERT dim: {emb_a.shape[1]}")
    else:
        emb_a = emb_b = None

    # Projector management
    proj_path = projector_model_path(data_hash)
    projs_path = projections_path(data_hash)

    st.header("Train projector ANN (contrastive)")
    st.write("Training trains ANN projector to map SBERT embeddings to projection space using NT-Xent (in-batch negatives).")

    if os.path.exists(proj_path) and 'projector_loaded' not in st.session_state:
        st.sidebar.success(f"Found saved projector: {proj_path}")
        if st.sidebar.button("Load saved projector"):
            try:
                projector = tf.keras.models.load_model(proj_path)
                st.session_state['projector'] = projector
                st.session_state['projector_loaded'] = True
                st.success("Projector loaded.")
            except Exception as e:
                st.error(f"Failed to load projector: {e}")

    if st.button("Build projector (not train)"):
        if emb_a is None:
            st.error("No embeddings available. Encode views first.")
        else:
            try:
                hiddens = [hidden1] + ([hidden2] if hidden2 > 0 else [])
                projector = build_projector(input_dim=emb_a.shape[1], proj_dim=int(proj_dim), hidden_sizes=hiddens, dropout=float(dropout))
                st.session_state['projector'] = projector
                st.success("Projector model built and stored in session.")
            except Exception as e:
                st.error(f"Failed to build projector: {e}")

    if st.button("Train projector now"):
        if emb_a is None:
            st.error("No embeddings available. Encode views first.")
        else:
            projector = st.session_state.get('projector') or build_projector(input_dim=emb_a.shape[1], proj_dim=int(proj_dim), hidden_sizes=[hidden1] + ([hidden2] if hidden2 > 0 else []), dropout=float(dropout))
            st.session_state['projector'] = projector
            with st.spinner("Training projector (contrastive) ..."):
                try:
                    proj_model, history = train_projector(
                        projector,
                        emb_a.astype(np.float32),
                        emb_b.astype(np.float32),
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        lr=float(lr),
                        temperature=float(temperature),
                        val_split=0.05,
                        patience=5,
                        save_path=proj_path
                    )
                    st.session_state['projector'] = proj_model
                    st.success("Training finished. Projector saved (best) if possible.")
                    st.session_state['proj_history'] = history
                except Exception as e:
                    st.error(f"Projector training failed: {type(e).__name__}: {e}")

    if st.button("Compute & save movie projections"):
        if 'projector' not in st.session_state:
            st.error("No projector in session. Build or load projector first.")
        elif 'emb_a' not in st.session_state:
            st.error("No embeddings. Encode views first.")
        else:
            projector = st.session_state['projector']
            with st.spinner("Computing projections for all movies..."):
                try:
                    base_embeddings = st.session_state['emb_a'].astype(np.float32)
                    movie_projs = compute_movie_projections(projector, base_embeddings, batch_size=1024)
                    np.save(projs_path, movie_projs)
                    st.session_state['movie_projs'] = movie_projs
                    st.success(f"Computed projections and saved to {projs_path}")
                except Exception as e:
                    st.error(f"Failed to compute projections: {type(e).__name__}: {e}")

    if 'movie_projs' not in st.session_state:
        if os.path.exists(projs_path) and not force_reproj:
            try:
                st.session_state['movie_projs'] = np.load(projs_path)
                st.success("Loaded cached movie projections from disk.")
            except Exception:
                pass

    st.header("Get recommendations (inference)")
    user_input = st.text_input("Describe what you want to watch (e.g. 'serial thriller about secret agent')", "")

    if st.button("Recommend") and user_input.strip():
        if 'projector' not in st.session_state:
            if os.path.exists(proj_path):
                try:
                    st.session_state['projector'] = tf.keras.models.load_model(proj_path)
                except Exception:
                    st.error("Projector not in session and failed to load from disk. Train or load projector first.")
                    st.stop()
            else:
                st.error("No projector available. Train projector first.")
                st.stop()

        if 'movie_projs' not in st.session_state:
            if os.path.exists(projs_path):
                try:
                    st.session_state['movie_projs'] = np.load(projs_path)
                except Exception:
                    st.error("Movie projections not in session and failed to load from disk. Compute projections first.")
                    st.stop()
            else:
                st.error("No movie projections available. Compute & save projections first.")
                st.stop()

        projector = st.session_state['projector']
        movie_projs = st.session_state['movie_projs']
        sbert = st.session_state.get('sbert_model') or load_sbert(model_name)

        try:
            q_emb = sbert.encode([user_input], convert_to_numpy=True)
            q_emb = q_emb / np.linalg.norm(q_emb, ord=2, axis=1, keepdims=True)
            q_emb = q_emb.astype(np.float32)
        except Exception as e:
            st.error(f"Failed to encode query with SBERT: {type(e).__name__}: {e}")
            st.stop()

        try:
            q_proj = projector(q_emb, training=False).numpy()
            sims = np.dot(q_proj, movie_projs.T)[0]
            top_idx = np.argsort(-sims)[:top_k]
            top_scores = sims[top_idx]
        except Exception as e:
            st.error(f"Inference failed: {type(e).__name__}: {e}")
            st.stop()

        results = []
        for idx, score in zip(top_idx, top_scores):
            row = movies_df.iloc[int(idx)]
            results.append({
                "title": row.get('title', ''),
                "release_date": row.get('release_date', ''),
                "overview": row.get('overview', ''),
                "genres": ", ".join(row.get('genres_list', [])),
                "cast": ", ".join(row.get('cast_list', [])),
                "score": float(score)
            })

        if not results:
            st.info("No recommendations produced. Try another query or retrain projector.")
        else:
            st.subheader(f"Top {len(results)} recommendations for: \"{user_input}\"")
            for i, r in enumerate(results, 1):
                st.markdown(f"**{i}. {r['title']}**")
                st.write(f"Score: {r['score']:.6f}")
                st.write(f"Release date: {r.get('release_date','N/A')}")
                st.write(r.get('overview',''))
                st.write(f"Genres: {r.get('genres','N/A')}")
                st.write(f"Cast: {r.get('cast','N/A')}")
                st.markdown("---")

    st.sidebar.markdown("---")
    st.sidebar.write("Tips:")
    st.sidebar.write("- For a quick baseline: after encode, try SBERT cosine nearest neighbors (np.dot) for comparison.")
    st.sidebar.write("- In-batch negatives work better with a large batch size if memory allows.")
    st.sidebar.write("- If dataset is large, use FAISS or hnswlib for retrieval on movie projections.")
    st.sidebar.write("- Always keep embeddings and projections L2-normalized during training and inference.")

if __name__ == "__main__":
    main()

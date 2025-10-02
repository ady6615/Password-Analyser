# password_checker_fixed.py 1
import os
import random
import pickle
from collections import Counter
import math
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------ CONFIG ------------------
DATA_FILE = "rockyou.txt"           # <-- put your dataset path/name here
FREQ_PKL  = "freq_full.pkl"
MODEL_PKL = "password_model.pkl"
VECT_PKL  = "vectorizer.pkl"

TRAIN_SAMPLE_SIZE = 500_000         # how many UNIQUE passwords to use for training
MAX_FEATURES = 5000                 # for char n-grams
NGRAM_RANGE = (1, 3)                # character ngrams (1..3)
COMMONNESS_PERCENTILE = 90          # top X% counts considered "common" during training
RANDOM_SEED = 42
# --------------------------------------------

def build_full_frequency(path):
    """Stream the password file and build a full Counter -> saved to disk."""
    if os.path.exists(FREQ_PKL):
        print("Loading existing full-frequency dict from", FREQ_PKL)
        with open(FREQ_PKL, "rb") as f:
            freq = pickle.load(f)
        total = sum(freq.values())
        return freq, total

    print("Building full-frequency dict (this may take a minute)...")
    cnt = Counter()
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            pw = line.rstrip("\n\r")
            if pw:
                cnt[pw] += 1
            # occasional progress print
            if i % 2_000_000 == 0:
                print(f"  read {i:,} lines, unique so far: {len(cnt):,}")
    total = sum(cnt.values())
    print(f"Done. Total passwords: {total:,}, unique passwords: {len(cnt):,}")
    with open(FREQ_PKL, "wb") as f:
        pickle.dump(cnt, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved full-frequency dict to", FREQ_PKL)
    return cnt, total

def train_model_from_freq(freq: Counter):
    """Train model on a sample of unique passwords (labels: top COMMONNESS_PERCENTILE as 'common')."""
    if os.path.exists(MODEL_PKL) and os.path.exists(VECT_PKL):
        print("Found saved model & vectorizer. Loading them.")
        with open(MODEL_PKL, "rb") as f:
            model = pickle.load(f)
        with open(VECT_PKL, "rb") as f:
            vect = pickle.load(f)
        return model, vect

    print("Preparing training sample from full frequency dictionary...")
    unique_pw = list(freq.keys())
    n_unique = len(unique_pw)
    sample_size = min(TRAIN_SAMPLE_SIZE, n_unique)
    random.seed(RANDOM_SEED)
    sample_pw = random.sample(unique_pw, sample_size)
    # Determine threshold for label: top COMMONNESS_PERCENTILE percent by count
    counts = np.array([freq[p] for p in sample_pw])
    thresh = np.percentile(counts, COMMONNESS_PERCENTILE)
    y = (counts >= thresh).astype(int)  # 1 = "common", 0 = "not common"
    print(f"Training sample size: {sample_size:,}. Commonness threshold (count) = {thresh:.0f}")

    print("Fitting char n-gram vectorizer...")
    vect = CountVectorizer(analyzer="char", ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
    X = vect.fit_transform(sample_pw)  # sparse

    print("Training LogisticRegression (predicts prob of 'common')...")
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X, y)

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(VECT_PKL, "wb") as f:
        pickle.dump(vect, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model & vectorizer saved.")
    return model, vect

# Utility formatting
def one_in(prob, cap=int(1e12)):
    if prob <= 0:
        return f"> {cap:,}"
    est = 1.0 / prob
    return f"{int(est):,}" if est < cap else f"> {cap:,}"

def sci(prob):
    return f"{prob:.3e}"

# Check function
def check_password(pw: str, freq: Counter, total: int, model, vect):
    # Exact lookup using FULL freq dict
    if pw in freq:
        c = freq[pw]
        prob = c / total
        print(f"\n⚠️  Exact match: password found in corpus")
        print(f"    Count: {c:,}")
        print(f"    Probability: {sci(prob)}  (~{prob*1_000_000:.3f} per million, ~1 in {one_in(prob)})")
        return

    # Not found — predict with ML
    Xnew = vect.transform([pw])
    # model.predict_proba -> [prob_not_common, prob_common] if trained that way
    try:
        p_common = float(model.predict_proba(Xnew)[0][1])
    except Exception:
        # fallback: if model doesn't support predict_proba (e.g., some classifiers), use decision_function
        df = model.decision_function(Xnew)
        p_common = 1/(1+math.exp(-float(df)))

    print(f"\n✅  Not found in corpus (no exact match).")
    print(f"    Predicted probability it's 'common': {p_common:.4f} (0..1)")
    print(f"    Interpret as: higher => looks more like common passwords")

# ----------- Main -------------
def main():
    if not os.path.exists(DATA_FILE):
        print("ERROR: dataset file not found at:", DATA_FILE)
        return

    # 1) build or load full frequency dict (for exact lookups)
    freq, total = build_full_frequency(DATA_FILE)

    # 2) train or load model (trained on a sample but full freq remains available)
    model, vect = train_model_from_freq(freq)

    # 3) interactive loop
    print("\nReady. Type passwords to check (type 'exit' to quit).")
    while True:
        try:
            pw = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            break
        if not pw:
            continue
        if pw.lower() == "exit":
            break
        check_password(pw, freq, total, model, vect)

if __name__ == "__main__":
    main()

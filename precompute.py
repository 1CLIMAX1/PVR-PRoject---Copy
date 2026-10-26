# precompute.py
import pickle, numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df         = pickle.load(open("movies.pkl", "rb"))
tfidf      = pickle.load(open("tfidf.pkl", "rb"))

TOP_K      = 20
BATCH_SIZE = 500
n          = tfidf.shape[0]
neighbours = {}

# Pre-extract for speed
ratings = df['averageRating'].to_numpy()
votes   = df['numVotes'].to_numpy()

# Normalise votes to 0-1 range so it doesn't overpower rating
max_votes = votes.max()

for start in range(0, n, BATCH_SIZE):
    end   = min(start + BATCH_SIZE, n)
    batch = cosine_similarity(tfidf[start:end], tfidf)
    for local_i, row in enumerate(batch):
        global_i      = start + local_i
        row[global_i] = -1  # exclude self

        # ── Blend: similarity + rating + popularity ──
        scores = (
            0.85 * row +
            0.10 * (ratings / 10.0) +          # normalise to 0-1
            0.05 * (votes / max_votes)          # normalise to 0-1
        )

        top_k = np.argpartition(scores, -TOP_K)[-TOP_K:]
        neighbours[global_i] = top_k[np.argsort(scores[top_k])[::-1]].tolist()

    print(f"{end}/{n}", flush=True)

pickle.dump(neighbours, open("neighbours.pkl", "wb"))
print("Done.")
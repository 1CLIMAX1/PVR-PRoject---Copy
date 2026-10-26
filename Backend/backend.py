from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from difflib import get_close_matches
import os
import pickle

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

API_KEY = os.getenv("OMDB_API_KEY")

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "movies_data.pkl"), "rb") as f:
    movies_data = pickle.load(f)

TITLES = movies_data["titles"]
TITLES_LOWER = movies_data["titles_lower"]
VOTES = movies_data["votes"]
RATINGS = movies_data["ratings"]

EXACT_TITLE_INDEX = {}
for i, title_lower in enumerate(TITLES_LOWER):
    current = EXACT_TITLE_INDEX.get(title_lower)
    if current is None or (VOTES[i], RATINGS[i]) > (VOTES[current], RATINGS[current]):
        EXACT_TITLE_INDEX[title_lower] = i

with open(os.path.join(BASE_DIR, "neighbours_data.pkl"), "rb") as f:
    neighbours_data = pickle.load(f)

NEIGHBOURS_OFFSETS = neighbours_data["offsets"]
NEIGHBOURS_FLAT = neighbours_data["flat"]

print(f"Ready — {len(TITLES)} movies indexed.", flush=True)


def _get_neighbours(idx: int):
    start = NEIGHBOURS_OFFSETS[idx]
    end = NEIGHBOURS_OFFSETS[idx + 1]
    return NEIGHBOURS_FLAT[start:end]


# ─── OMDB ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=128)
def get_movie_details(title: str) -> dict:
    _default = {
        "poster":    "https://placehold.co/200x300/1a1a1a/666?text=No+Poster",
        "rating":    "N/A",
        "director":  "N/A",
        "actors":    "N/A",
        "released":  "N/A",
        "genre":     "N/A",
        "boxoffice": "N/A",
    }
    if not API_KEY:
        return _default
    try:
        url  = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey={API_KEY}"
        data = requests.get(url, timeout=5).json()
        if data.get("Response") != "True":
            return _default

        poster = data.get("Poster") or ""
        if poster in ("", "N/A"):
            poster = _default["poster"]

        actors = data.get("Actors", "N/A")
        if actors != "N/A":
            actors = ", ".join(actors.split(",")[:3])

        return {
            "poster":    poster,
            "rating":    data.get("imdbRating", "N/A"),
            "director":  data.get("Director",   "N/A"),
            "actors":    actors,
            "released":  data.get("Released",   "N/A"),
            "genre":     data.get("Genre",      "N/A"),
            "boxoffice": data.get("BoxOffice",  "N/A"),
        }
    except Exception:
        return _default


# ─── Title matching ───────────────────────────────────────────────────────────
def _find_movie_index(title: str):
    t = title.strip().lower()

    exact_index = EXACT_TITLE_INDEX.get(t)
    if exact_index is not None:
        return exact_index

    candidates = [i for i, title_lower in enumerate(TITLES_LOWER) if t in title_lower]
    if candidates:
        starts = [i for i in candidates if TITLES_LOWER[i].startswith(t)]
        pool = starts if starts else candidates
        return max(pool, key=lambda i: (VOTES[i], RATINGS[i]))

    close = get_close_matches(t, TITLES_LOWER, n=1, cutoff=0.55)
    if close:
        return EXACT_TITLE_INDEX.get(close[0])

    return None


# ─── Recommendation logic ─────────────────────────────────────────────────────
@lru_cache(maxsize=128)
def recommend(title: str) -> list:
    key = title.strip().lower()
    idx = _find_movie_index(title)
    if idx is None:
        return []

    neighbours = _get_neighbours(idx)
    top = [i for i in neighbours if VOTES[i] > 0 and RATINGS[i] > 0][:5]
    if not top:
        top = neighbours[:5]

    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(get_movie_details, TITLES[i]): i
            for i in top
        }
        for future in as_completed(futures):
            i = futures[future]
            details = future.result()
            results.append({
                "title": TITLES[i],
                "poster": details["poster"],
                "rating": details["rating"],
                "director": details["director"],
                "actors": details["actors"],
                "released": details["released"],
                "genre": details["genre"],
                "boxoffice": details["boxoffice"],
            })

    return results


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/ping")
def ping():
    return "ok", 200

@app.route("/recommend", methods=["POST", "OPTIONS"])
def recommend_api():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json()
    if not data or "movie" not in data:
        return jsonify({"error": "No movie provided"}), 400

    try:
        results = recommend(data["movie"])
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
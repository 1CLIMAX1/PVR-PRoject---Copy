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

GENRE_MASKS = movies_data["genre_masks"]
GENRES = movies_data["genres"]
PAGE_SIZE = 5

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

@lru_cache(maxsize=64)
def _search(q, genre, sort):
    t = q.strip().lower()
    gbit = 1 << GENRES.index(genre) if genre in GENRES else 0
    hits = []
    for i in range(len(TITLES)):
        if gbit and not GENRE_MASKS[i] & gbit:
            continue
        if sort == "rating" and VOTES[i] < 1.0:   # z-score: above-average votes only
            continue
        rank = 0
        if t:
            tl = TITLES_LOWER[i]
            if t not in tl:
                continue
            rank = 0 if tl == t else 1 if tl.startswith(t) else 2
        hits.append((rank, i))

    keys = {
        "rating": lambda h: (-RATINGS[h[1]], -VOTES[h[1]]),
        "title":  lambda h: TITLES_LOWER[h[1]],
    }
    hits.sort(key=keys.get(sort, lambda h: (h[0], -VOTES[h[1]])))
    return [i for _, i in hits]


def _genres_of(i):
    m = GENRE_MASKS[i]
    return [g for b, g in enumerate(GENRES) if m >> b & 1]


@app.route("/genres")
def genres():
    return jsonify(GENRES)


@app.route("/details")
def details():
    return jsonify(get_movie_details(request.args.get("title", "")))

@app.route("/search")

def search():
    a = request.args
    try:
        page = max(1, int(a.get("page", 1)))
    except ValueError:
        page = 1
    ids = _search(a.get("q", ""), a.get("genre", ""), a.get("sort", "popularity"))
    chunk = ids[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return jsonify({
        "total": len(ids),
        "page": page,
        "pages": max(1, -(-len(ids) // PAGE_SIZE)),
        "results": [{"title": TITLES[i], "genres": _genres_of(i)} for i in chunk],
    })


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

@lru_cache(maxsize=256)
def get_full_details(title: str) -> dict:
    if not API_KEY:
        return {"Response": "False"}
    try:
        return requests.get(
            "http://www.omdbapi.com/",
            params={"t": title, "plot": "full", "apikey": API_KEY},
            timeout=5,
        ).json()
    except Exception:
        return {"Response": "False"}


@app.route("/movie")
def movie():
    return jsonify(get_full_details(request.args.get("title", "")))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
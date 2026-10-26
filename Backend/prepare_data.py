import os
import pickle
from array import array

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVIES_FILE = os.path.join(BASE_DIR, "movies.pkl")
NEIGHBOURS_FILE = os.path.join(BASE_DIR, "neighbours.pkl")
COMPACT_MOVIES_FILE = os.path.join(BASE_DIR, "movies_data.pkl")
COMPACT_NEIGHBOURS_FILE = os.path.join(BASE_DIR, "neighbours_data.pkl")


def build_compact_movies():
    with open(MOVIES_FILE, "rb") as f:
        df = pickle.load(f)

    titles = df["primaryTitle"].astype(str).tolist()
    titles_lower = [title.strip().lower() for title in titles]
    votes = array("f", df["numVotes"].fillna(0).astype("float32").tolist())
    ratings = array("f", df["averageRating"].fillna(0).astype("float32").tolist())

    movies_data = {
        "titles": titles,
        "titles_lower": titles_lower,
        "votes": votes,
        "ratings": ratings,
    }

    with open(COMPACT_MOVIES_FILE, "wb") as f:
        pickle.dump(movies_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote compact movies data: {COMPACT_MOVIES_FILE}")


def build_compact_neighbours():
    with open(NEIGHBOURS_FILE, "rb") as f:
        neighbours = pickle.load(f)

    flat = array("I")
    offsets = array("I")
    total = 0

    for idx in range(len(neighbours)):
        offsets.append(total)
        neighbours_list = neighbours.get(idx, [])
        flat.extend(neighbours_list)
        total += len(neighbours_list)
    offsets.append(total)

    neighbours_data = {"offsets": offsets, "flat": flat}
    with open(COMPACT_NEIGHBOURS_FILE, "wb") as f:
        pickle.dump(neighbours_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote compact neighbours data: {COMPACT_NEIGHBOURS_FILE}")


if __name__ == "__main__":
    build_compact_movies()
    build_compact_neighbours()

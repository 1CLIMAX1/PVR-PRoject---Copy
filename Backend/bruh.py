import pickle
df = pickle.load(open("movies.pkl", "rb"))
print(df.head(3).to_dict("records"))
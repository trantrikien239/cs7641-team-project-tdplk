from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

data_path = "../../data/train_tokenized.csv"

data = pd.read_csv(data_path) # Palash's file
data.drop(columns=["Unnamed: 0"], inplace=True)
data["full_text"] = data["full_text"].apply(lambda x: x.strip())
tasks = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
data["holistic_score"] = data[tasks].mean(axis=1)

# Load the model
smodel = SentenceTransformer('all-distilroberta-v1')
X_embeddings = smodel.encode(data["full_text"].values, show_progress_bar=True)
X_embeddings_df = pd.concat([data[["text_id"]], pd.DataFrame(X_embeddings)], axis=1)

# Save the embeddings
X_embeddings_df.to_parquet("../../data/X_embeddings.parquet")

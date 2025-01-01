import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
import numpy as np
import nltk
from nltk.corpus import stopwords

turkish_stop_words = stopwords.words('turkish')
df = pd.read_csv('books.csv')
df = df[['id', 'author', 'book_type', 'explanation']]
df['author'] = df['author'].fillna('unknown')
df['book_type'] = df['book_type'].fillna('unknown')
df['explanation'] = df['explanation'].fillna('')

tfidf = TfidfVectorizer(stop_words=turkish_stop_words, max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['explanation'])

author_similarity = df['author'].values[:, None] == df['author'].values[None, :]
author_similarity = author_similarity.astype(float)

type_similarity = df['book_type'].values[:, None] == df['book_type'].values[None, :]
type_similarity = type_similarity.astype(float)

explanation_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

weight_author = 0.2
weight_type = 0.2
weight_explanation = 0.6

total_similarity = (weight_author * author_similarity) + \
                   (weight_type * type_similarity) + \
                   (weight_explanation * explanation_similarity)

indices = pd.Series(df.index, index=df['id']).drop_duplicates()
def get_recommendations(book_id, total_similarity=total_similarity, indices=indices):
    if book_id not in indices:
        return "Kitap ID bulunamadı."
    idx = indices[book_id]
    sim_scores = list(enumerate(total_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] 
    book_indices = [i[0] for i in sim_scores]
    return df['id'].iloc[book_indices].tolist()
def get_recommendationsForUser(book_ids, total_similarity=total_similarity, indices=indices, top_n=10):
    valid_indices = []
    for book_id in book_ids:
        if book_id in indices:
            valid_indices.append(indices[book_id])
        else:
            print(f"Kitap ID bulunamadı: {book_id}")
    if not valid_indices:
        return "Hiçbir geçerli kitap ID'si bulunamadı."
    combined_similarity = np.sum(total_similarity[valid_indices], axis=0)
    combined_similarity[valid_indices] = 0
    recommended_indices = combined_similarity.argsort()[-top_n:][::-1]
    recommendations = df['id'].iloc[recommended_indices].tolist()
    return recommendations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
import numpy as np
import nltk
from nltk.corpus import stopwords
import joblib

class BookRecommender:
    def __init__(self, csv_path, max_features=5000):
        # Load data
        self.df = pd.read_csv(csv_path)
        self.df = self.df[['id', 'author', 'book_type', 'explanation']]
        self.df['author'] = self.df['author'].fillna('unknown')
        self.df['book_type'] = self.df['book_type'].fillna('unknown')
        self.df['explanation'] = self.df['explanation'].fillna('')

        self.turkish_stop_words = stopwords.words('turkish')

        self.vectorizer = TfidfVectorizer(stop_words=self.turkish_stop_words, max_features=max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['explanation'])
        self.author_similarity = self._compute_author_similarity()
        self.type_similarity = self._compute_type_similarity()
        self.explanation_similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.weight_author = 0.2
        self.weight_type = 0.2
        self.weight_explanation = 0.6
        self.total_similarity = (self.weight_author * self.author_similarity) + \
                                (self.weight_type * self.type_similarity) + \
                                (self.weight_explanation * self.explanation_similarity)

        # Create indices for quick lookup
        self.indices = pd.Series(self.df.index, index=self.df['id']).drop_duplicates()

    def _compute_author_similarity(self):
        author_matrix = self.df['author'].values[:, None] == self.df['author'].values[None, :]
        return author_matrix.astype(float)

    def _compute_type_similarity(self):
        type_matrix = self.df['book_type'].values[:, None] == self.df['book_type'].values[None, :]
        return type_matrix.astype(float)

    def get_recommendations(self, book_id, top_n=10):
        if book_id not in self.indices:
            return "Kitap ID bulunamadı."
        idx = self.indices[book_id]
        sim_scores = list(enumerate(self.total_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1] 
        book_indices = [i[0] for i in sim_scores]
        return self.df['id'].iloc[book_indices].tolist()

    def get_recommendationsForUser(self, book_ids, top_n=10):
        valid_indices = []
        for book_id in book_ids:
            if book_id in self.indices:
                valid_indices.append(self.indices[book_id])
            else:
                print(f"Kitap ID bulunamadı: {book_id}")
        if not valid_indices:
            return "Hiçbir geçerli kitap ID'si bulunamadı."
        combined_similarity = np.sum(self.total_similarity[valid_indices], axis=0)
        combined_similarity[valid_indices] = 0 
        recommended_indices = combined_similarity.argsort()[-top_n:][::-1]
        recommendations = self.df['id'].iloc[recommended_indices].tolist()
        return recommendations

    def save_model(self, filepath):
        # Define what to save
        model_data = {
            'df': self.df,
            'vectorizer': self.vectorizer,
            'total_similarity': self.total_similarity,
            'indices': self.indices,
            'weights': {
                'author': self.weight_author,
                'type': self.weight_type,
                'explanation': self.weight_explanation
            }
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        model_data = joblib.load(filepath)
        obj = cls.__new__(cls)
        obj.df = model_data['df']
        obj.vectorizer = model_data['vectorizer']
        obj.total_similarity = model_data['total_similarity']
        obj.indices = model_data['indices']
        weights = model_data['weights']
        obj.weight_author = weights['author']
        obj.weight_type = weights['type']
        obj.weight_explanation = weights['explanation']
        return obj
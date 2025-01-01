from  book_recommender import BookRecommender


def train():
    recommender = BookRecommender(csv_path='books.csv', max_features=5000)
    recommender.save_model('book_recommender_model.joblib')
if __name__ == '__main__':
    train()
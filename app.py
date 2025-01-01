from flask import Flask, request, jsonify
from flask_cors import CORS
from book_recommender import BookRecommender
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model/book_recommender_model.joblib'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train and save the model first.")

recommender = BookRecommender.load_model(MODEL_PATH)

@app.route('/recommend', methods=['GET'])
def recommend():
    book_id = request.args.get('id')
    if book_id is None:
        return jsonify({'error': 'Kitap ID\'si belirtilmedi.'}), 400

    recommendations = recommender.get_recommendations(book_id)
    if isinstance(recommendations, str):
        return jsonify({'error': recommendations}), 400
    return jsonify({'recommendations': recommendations})

@app.route('/recommends', methods=['GET'])
def recommends():
    ids = request.args.get('ids')
    if not ids:
        return jsonify({'error': 'Hiçbir kitap ID\'si belirtilmedi.'}), 400
    try:
        book_ids = [id_str for id_str in ids.split(',')]
    except ValueError:
        return jsonify({'error': 'Kitap ID\'leri geçersiz formatta.'}), 400

    recommendations = recommender.get_recommendationsForUser(book_ids)
    if isinstance(recommendations, str):
        return jsonify({'error': recommendations}), 400
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
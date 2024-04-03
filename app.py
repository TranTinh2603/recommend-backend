from flask import Flask, jsonify
from flask_cors import CORS
from data_recommendation import prepare_data, recommend_items
from data_similar import prepare_data_similar, get_similar_items


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3001"}})

@app.route('/api/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    ratings_matrix, item_similarity_matrix = prepare_data()
    top_n_items = recommend_items(user_id, ratings_matrix, item_similarity_matrix)
    recommendations = [str(ratings_matrix.columns[item_idx]) for item_idx in top_n_items]
    return jsonify({"recommendations": recommendations})

@app.route('/api/similars/<book_id>', methods=['GET'])
def get_similars(book_id):
    cosine_similarities, filtered_data = prepare_data_similar()
    similar_items = get_similar_items(book_id, cosine_similarities, filtered_data)
    print("\nCác mục tương tự nhất cho mục đầu tiên:")
    similars = []
    for item in similar_items:
        similars.append(item[0]['documentId'])
        print("Độ tương tự", str(item[1]))
    return jsonify({"similars": similars})

if __name__ == '__main__':
    app.run(debug=True)

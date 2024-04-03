import pymongo
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from scipy.spatial.distance import cosine


# Khởi tạo ứng dụng Flask
app = Flask(__name__)
# Cho phép CORS để chấp nhận các yêu cầu từ nguồn cụ thể
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3001"}})

# Định nghĩa biến global cho dữ liệu cần sử dụng
ratings_matrix = None
item_similarity = None

# Kết nối đến cơ sở dữ liệu MongoDB và chuẩn bị dữ liệu
def prepare_data():
    global ratings_matrix, item_similarity_matrix
    # Kết nối đến cơ sở dữ liệu MongoDB
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["social-network"]
    mycol = mydb["reviews"]

    # Lấy dữ liệu từ collection "reviews" và lưu vào danh sách reviews
    reviews = []
    for x in mycol.find():
        reviews.append(x)

    # Tạo một từ điển để lưu trữ đánh giá của từng người dùng
    user_reviews = {}
    for review in reviews:
        user_id = review["userId"]
        if user_id not in user_reviews:
            user_reviews[user_id] = []
        user_reviews[user_id].append(review)

    # Chuẩn bị dữ liệu cho việc xây dựng ma trận đánh giá
    cleaned_data = []
    for user_id, reviews_list in user_reviews.items():
        for review in reviews_list:
            cleaned_item = {
                "userId": review["userId"],
                "bookId": review["bookId"],
                "rating": review["rating"]
            }
            cleaned_data.append(cleaned_item)

    # Lưu dữ liệu đã làm sạch vào file CSV
    csv_file = "reviews.csv"
    fields = ["userId", "bookId", "rating"]
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in cleaned_data:
            writer.writerow(item)

    # Đọc dữ liệu từ file CSV vào DataFrame và xây dựng ma trận đánh giá
    ratings_data = pd.read_csv('reviews.csv')
    ratings_matrix = ratings_data.pivot_table(index='userId', columns='bookId', values='rating', fill_value=0)
    # print(ratings_data)

   
    # Tính toán độ tương tự giữa các mặt hàng bằng cosine similarity
    item_similarity = cosine_similarity(ratings_matrix.T)  # Transpose of the ratings_matrix to get item-item similarity
    item_similarity_matrix = pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)

def predict_rating(user_id, item_id, ratings_matrix, item_similarity_matrix):
    similar_items = item_similarity_matrix[item_id]
    user_ratings = ratings_matrix.loc[user_id, : ]
    similar_ratings = user_ratings[similar_items.index]
    print("Tong cua", similar_items,"*", similar_ratings )
    weighted_sum = np.sum(similar_items * similar_ratings)
    print("Weighted sum", weighted_sum)
    sum_of_weights = np.sum(similar_items)
    print("Sum of weights", sum_of_weights)
    if sum_of_weights != 0:
        predicted_rating = weighted_sum / sum_of_weights
    else:
        predicted_rating = None
    return predicted_rating


def predict_ratings_for_user(user_id, ratings_matrix, item_similarity_matrix):
    predicted_ratings = []
    for item_id in ratings_matrix.columns:
        if pd.notnull(ratings_matrix.loc[user_id, item_id]):
            predicted_rating = predict_rating(user_id, item_id, ratings_matrix, item_similarity_matrix)
            predicted_ratings.append(predicted_rating)

    return predicted_ratings

# Hàm gợi ý các mặt hàng cho một người dùng cụ thể
def recommend_items(user_id, n=10):
    global ratings_matrix, item_similarity_matrix
    
    
    #Trích xuất hàng tương ứng với user_id từ ma trận đánh giá và chuyển đổi thành một mảng một chiều
    user_ratings = ratings_matrix.loc[user_id].values.reshape(1, -1)
    
    # Tính toán dự đoán xếp hạng cho người dùng bằng cách nhân ma trận user_ratings với ma trận độ tương tự item_similarity
    # predicted_ratings = np.dot(user_ratings, item_similarity).flatten()
    # print(predicted_ratings)
    predicted_ratings = predict_ratings_for_user(user_id, ratings_matrix, item_similarity_matrix)
    
    # Tìm các mặt hàng đã được người dùng đánh giá
    rated_items = np.where(user_ratings.flatten() != 0)[0]
    
    # Đánh dấu các mặt hàng đã được đánh giá bằng -1 trong mảng predicted_ratings
    # Điều này để đảm bảo các mặt hàng đã được đánh giá không được gợi ý lại
    
    array_predicted_ratings = np.array(predicted_ratings)
    array_predicted_ratings[rated_items] = -1

    print(array_predicted_ratings)
    
    # Sắp xếp các mặt hàng dựa trên dự đoán xếp hạng từ cao đến thấp và chọn ra n mặt hàng có xếp hạng cao nhất
    top_n_items = np.argsort(array_predicted_ratings)[::-1][:n]
    print("top item",top_n_items)
    
    # Trả về chỉ số của các mặt hàng được gợi ý
    return top_n_items

# Hàm lấy các gợi ý cho một người dùng cụ thể và trả về dưới dạng JSON
@app.route('/api/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    global ratings_matrix
    prepare_data()
    top_n_items = recommend_items(user_id)
    recommendations = [str(ratings_matrix.columns[item_idx]) for item_idx in top_n_items]
    
    return jsonify({"recommendations": recommendations})

# Hàm chạy trước khi xử lý request đầu tiên


# Khởi chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)

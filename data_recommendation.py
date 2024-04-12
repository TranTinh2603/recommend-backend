import pymongo
import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

def prepare_data():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["social-network"]
    mycol = mydb["reviews"]

    reviews = []
    for x in mycol.find():
        reviews.append(x)

    user_reviews = {}
    for review in reviews:
        user_id = review["userId"]
        if user_id not in user_reviews:
            user_reviews[user_id] = []
        user_reviews[user_id].append(review)

    cleaned_data = []
    for user_id, reviews_list in user_reviews.items():
        for review in reviews_list:
            cleaned_item = {
                "userId": review["userId"],
                "bookId": review["bookId"],
                "rating": review["rating"]
            }
            cleaned_data.append(cleaned_item)

    csv_file = "recommends/reviews.csv"
    fields = ["userId", "bookId", "rating"]
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in cleaned_data:
            writer.writerow(item)

    ratings_data = pd.read_csv('recommends/reviews.csv')
    ratings_matrix = ratings_data.pivot_table(index='userId', columns='bookId', values='rating', fill_value=0)

    item_similarity = cosine_similarity(ratings_matrix.T)
    item_similarity_matrix = pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)

    print("ratings_matrix", ratings_matrix)
    print("item_similarity_matrix", item_similarity_matrix)
    
    return ratings_matrix, item_similarity_matrix

def predict_rating(user_id, item_id, ratings_matrix, item_similarity_matrix):
    similar_items = item_similarity_matrix[item_id]
    print("similar_items: ",similar_items.index)
    user_ratings = ratings_matrix.loc[user_id, :]
    print("user_ratings", user_ratings)
    similar_ratings = user_ratings[similar_items.index]
    print("similar_ratings", similar_ratings)
    similar_items_similar_ratings_sum = np.sum(similar_items * similar_ratings)
    print("similar_items_similar_ratings_sum", similar_items_similar_ratings_sum)
    sum_similar_item = np.sum(similar_items)
    print("sum_similar_item",sum_similar_item)
    if sum_similar_item != 0:
        predicted_rating = similar_items_similar_ratings_sum / sum_similar_item
    else:
        predicted_rating = None
    print("predicted_rating", predicted_rating)
    return predicted_rating

def predict_ratings_for_user(user_id, ratings_matrix, item_similarity_matrix):
    predicted_ratings = []
    for item_id in ratings_matrix.columns:
        if pd.notnull(ratings_matrix.loc[user_id, item_id]):
            predicted_rating = predict_rating(user_id, item_id, ratings_matrix, item_similarity_matrix)
            predicted_ratings.append(predicted_rating)
    return predicted_ratings

def recommend_items(user_id, ratings_matrix, item_similarity_matrix, n=10):
    user_ratings = ratings_matrix.loc[user_id].values.reshape(1, -1)
    predicted_ratings = predict_ratings_for_user(user_id, ratings_matrix, item_similarity_matrix)
    print("predicted_ratings",predicted_ratings)
    rated_items = np.where(user_ratings.flatten() != 0)[0]
    array_predicted_ratings = np.array(predicted_ratings)
    print("array_predicted_ratings", array_predicted_ratings)
    array_predicted_ratings[rated_items] = -1
    top_n_items = np.argsort(array_predicted_ratings)[::-1][:n]
    return top_n_items

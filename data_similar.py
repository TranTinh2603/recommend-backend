import io
import sys
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def prepare_data_similar():


    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["social-network"]
    mycol = mydb["books"]
    books = []
    for x in mycol.find():
        books.append(x)


    # Sử dụng biến filtered_data thay vì filtered_data
    filtered_data = [{"documentId": book['bookId'], "document": book['author'] + ',' + book['category']} for book in books]

    # In ra filtered_data
    print(filtered_data)
    documents = [item['document'] for item in filtered_data]
    print(documents)



    vectorizer = TfidfVectorizer()
    print(vectorizer)
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(tfidf_matrix)


    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Ma trận tương đồng cosine giữa các mục:")
    print(cosine_similarities)

    return cosine_similarities, filtered_data

def get_similar_items(item_index, cosine_similarities, documents):
    similar_items = []
    for i, item in enumerate(documents):
        if item['documentId'] != item_index:
            continue
        for j, similarity in enumerate(cosine_similarities[i]):
            print("similarity", similarity)
            if i != j and similarity > 0:
                print(documents[j])
                similar_items.append((documents[j], similarity))
    similar_items.sort(key=lambda x: x[1], reverse=True)
    return similar_items
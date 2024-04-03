import io
import sys
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Giải thuật 1


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["social-network"]
mycol = mydb["books"]

print(mycol)
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

similar_items = get_similar_items("B002", cosine_similarities, filtered_data)
print("\nCác mục tương tự nhất cho mục đầu tiên:")
for item in similar_items:
    print(item[0]['documentId'] + ' ' +str(item[1]))


# Giải thuật 2

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dữ liệu mẫu - ví dụ về các mục có thuộc tính tác giả và thể loại
data = [
    {"author": "Nguyễn Nhật Ánh", "genre": "Tiểu Thuyết"},
    {"author": "Nguyễn Nhật Ánh", "genre": "Thơ"},
    {"author": "Nguyễn Du", "genre": "Thơ"},
    {"author": "Nguyễn Nhật Ánh", "genre": "Tiểu Thuyết"},
    {"author": "Nam Cao", "genre": "Kinh tế"}
]

# Sử dụng DictVectorizer để biểu diễn các mục thành vector
vectorizer = DictVectorizer()
print(vectorizer)
X = vectorizer.fit_transform(data)
print(X)

# Tính toán ma trận tương đồng cosine giữa các mục
cosine_similarities = cosine_similarity(X, X)

# In ra ma trận tương đồng
print("Ma trận tương đồng cosine giữa các mục:")
print(cosine_similarities)

# Hàm gợi ý các mục tương tự nhất cho một mục cụ thể
def get_similar_items(item_index, cosine_similarities, data):
    similar_items = []
    for i, similarity in enumerate(cosine_similarities[item_index]):
        if i != item_index and similarity > 0:
            similar_items.append((data[i], similarity))
    similar_items.sort(key=lambda x: x[1], reverse=True)
    return similar_items

# Ví dụ: Gợi ý các mục tương tự nhất cho mục đầu tiên trong danh sách
similar_items = get_similar_items(0, cosine_similarities, data)
print("\nCác mục tương tự nhất cho mục đầu tiên:")
for item in similar_items:
    print(item)
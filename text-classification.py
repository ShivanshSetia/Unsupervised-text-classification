from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

file = open("docs","r")
document = file.read()
document = document.split(",")
print(type(document))


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(document)

def gen_model(X,n_clusters,init='k-means++',max_iter=100,n_init=1):
    model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter, n_init=n_init)
    model.fit(X)
    return model

model = gen_model(X,2)
# to get centroid and features

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

# Printing in the cluster they belong

for i in range(2):
    print(" Cluster : " + str(i)),
    for ind in order_centroids[i, :10]:
        print(terms[ind])

print("\n")
print("Prediction")
X = vectorizer.transform(["Nothing is easy in cricket. Maybe when you watch it on TV, it looks easy. But it is not. You have to use your brain and time the ball."])
predicted = model.predict(X)
print(predicted)

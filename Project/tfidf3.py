import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
text = ["This is the first document.","This document is the second document."]
tfidf = TfidfVectorizer(norm=None)
tfidf_matrix = tfidf.fit_transform(text).toarray()
print("tfidf.vocabulary_")
print(tfidf.vocabulary_)
print(tfidf_matrix)
columns=tfidf.get_feature_names()
print("columns")
print(columns)

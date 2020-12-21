import numpy as np
import pandas as pd
import itertools
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pickle

#Read the data
df=pd.read_csv(r'news.csv')

X = df['text']
y = df['label']

x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.2)

pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words = 'english')),
                    ('nbmodel', MultinomialNB())])

pipeline.fit(x_train, y_train)

pred = pipeline.predict(x_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol = pickle.HIGHEST_PROTOCOL)
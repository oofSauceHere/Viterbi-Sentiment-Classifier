class Word:
    def __init__(self, word, freq):
        self.word = word
        self.freq = freq
    def add(self):
        self.freq += 1

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.tsv', delimiter = '\t', quoting = 3)

# Creating the "keywords" list
import re
keywords = []
for i in range(0, 10662):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    for wd in review:
        isIn = False
        for w in keywords:
            if w.word == wd:
                w.add()
                isIn = True
                break
        if isIn == False:
            keywords.append(Word(wd, 1))

# Cleaning the texts
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
corpus = []
for i in range(0, 10662):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    lz = WordNetLemmatizer()
    review = [lz.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500, ngram_range = (1, 3))
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Fitting HMM to the Training set
from hmmlearn.hmm import GaussianHMM
classifier = GaussianHMM()
classifier.fit(X_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
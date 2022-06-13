class Word:
    def __init__(self, word, freq):
        self.word = word
        self.freq = freq
    def add(self):
        self.freq += 1

class Data_processor:
    def __init__(self, path):
        self.datal = []
        self.data = []
        self.file = open(path, 'r', encoding="utf8")
        for i in self.file.read().split("\n\n") :
            sentence = []
            unlabeled = []
            for j in i.split("\n"):
                if j != "":
                    unlabeled.append(j)
                    word = j.split(" ")[0].lower()
                    if len(word) > 5 and word[:4] == "http":
                        word = word[:4]
                    if len(j.split(" ")) >1:
                        sentence.append(word + " " + j.split(" ")[-1])
                    else:
                        sentence.append(word)
            if len(sentence) > 0:
                self.data.append(sentence)
                self.datal.append(unlabeled)
        self.file.close()

dev_data = Data_processor("D:\MachineLearning\Project\dev.in").data
training_data = Data_processor("train.txt").data

keywords = []
for i in range(len(dev_data)):
    for j in range(len(dev_data[i])-1):
         wd = dev_data[i][j]
         wd = wd.lower()
         isIn = False
         for w in keywords:
             if w.word == wd:
                 w.add()
                 isIn = True
                 break
         if isIn == False:
             keywords.append(Word(wd, 1))

# Cleaning the texts
from nltk.stem.snowball import SnowballStemmer
corpus = []
for i in range(len(dev_data)):
    review = []
    for j in range(len(dev_data[i])-1):
        ss = SnowballStemmer("english")
        review.append(ss.stem(dev_data[i][j]))
    corpus.append(' '.join(review))

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500, ngram_range = (1, 3))
X = cv.fit_transform(corpus).toarray()
y = dev_data.iloc[:, 1].values

X_train = train.txt.iloc[0].values
y_train = train.txt.iloc[1].values

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(dev_data.iloc[0].values)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dev_data.iloc[1].values, y_pred)
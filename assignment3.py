import numpy
import re
import nltk
from sklearn.datasets import load_files
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from nltk.corpus import abc
from gensim.models import Word2Vec
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from sklearn import svm


Reviews = load_files(r"C:\Users\dell\Desktop\ass3\txt_sentoken")

movies_reviews = [[]]*2000

x, y = Reviews.data, Reviews.target
#to transform the data

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

for i in range(0, len(x)):
    
    review = re.sub('\W', ' ', str(x[i]))
    review = re.sub('\s+[a-zA-Z]\s+', ' ', review)
    review = re.sub('\s+', ' ', review, flags=re.I)
    stopwords = word_tokenize(review)
    review_ = [word for word in stopwords if not word in all_stopwords]
    movies_reviews[i]=review_ 

model = gensim.models.Word2Vec(
        movies_reviews,
        vector_size=100,
        window=20,
        min_count=1,
        sg=1 #skip_gram
        )
#print(movies_reviews[0])

length = 0
def Sent_embedding(review):
    global model
    vec = 100*[0]
    vec_= []
    global length
    for word in review:
        vec += model.wv[word]
        length+=1
    for FV in vec:
        vec_.append(FV/length)
    return vec_

reviews=[]
for review in movies_reviews:
    review_=Sent_embedding(review)
    reviews.append(review_)

#split data
x_train, x_test, y_train, y_test = train_test_split(reviews, y, test_size=0.3, random_state=0)

#to classify the data (Forest)
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(x_train, y_train) 
pred = classifier.predict(x_test)

#SVM
SVM = svm.SVC(kernel='linear')
SVM.fit(x_train, y_train)
pred2 = SVM.predict(x_test)

#accuracy
print("the model's accuracy with Forest =",accuracy_score(y_test, pred))
print("the model's accuracy with SVM =",accuracy_score(y_test, pred2))

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:31:17 2022

@author: mohammedowais
"""
import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re 
from sklearn.feature_extraction.text import CountVectorizer

ds = pd.read_csv("SMSSpamCollection", sep='\t', names =["label","message"])
ps = PorterStemmer()
corpus =[]
for i in range(0, len(ds)):
    review = re.sub('[^a-zA-Z]',' ', ds['message'][i])
    review = review.lower()
    review = review.split()
    review =[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features =5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(ds['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=0)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model =MultinomialNB().fit(X_train,y_train)
y_pred =spam_detect_model.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
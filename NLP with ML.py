#import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read the data set
data=pd.read_csv(r"C:\Users\sunil\Desktop\DK\vs code\sentiment prediction by text\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

import re
import nltk
from nltk.corpus import stopwords # for stopwords
from nltk.stem.porter import PorterStemmer # for stem the words

# blank cor[]
corpus=[]

# take to proper format
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', data["Review"][i])
    review = review.lower()
    review = review.split()
    ps=PorterStemmer()
    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   
    review = ' '.join(review)
    corpus.append(review)
    
  
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)

#4 logostic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

# predict
y_pred=classifier.predict(x_test)

#accuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

# bais variance
bais=classifier.score(x_train,y_train)
var=classifier.score(x_test,y_test)
print("bais:-", bais)
print("variance:-",var)

import pickle
filename="model.pkl"
with open (filename,"wb") as file:
    pickle.dump(classifier,file)
    
#TfidfVectorizer file
tfidf_filename = 'tfidf.pkl'
with open(tfidf_filename, 'wb') as scaler_file:
    pickle.dump(cv, scaler_file)
   
import os
os.getcwd()

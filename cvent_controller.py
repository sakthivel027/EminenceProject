#cvent_controller.py

import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.externals import joblib 


#importing Dataset
dataset = pd.read_csv('train.csv')
dataset.head()
#Visualization of data
dataset.label.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "green"])
df= pd.read_csv('test.csv')


#Cleaning of Train dataset

corpus = []
for i in range(0,10000):
    #Selecting the useful things
    review=re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",dataset['tweet'][i])
    #Changing case of alpahbe
    review = review.lower()
    #Splitting the sentence
    review = review.split()
    #Removing unuseful words( like - is,the,this)
    review = [word for word in review if not word in stopwords.words('english')]
    #stemming(Finding root word from word by removing root word)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    review=" ".join(review)  
    corpus.append(review)
    
#Cleaning of Test dataset    
testing = []
for i in range(0,3062):
    review = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",df['tweet'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in stopwords.words('english')]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    review=" ".join(review)  
    testing.append(review)    



















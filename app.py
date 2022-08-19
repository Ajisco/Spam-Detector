import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, render_template

app= Flask(__name__)

data=pd.read_csv('spam.csv')
data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
clf.fit(X_train,y_train)




@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def index():
    data= str(request.form['text'])
    data=[data]
    #arr = np.array([[data]])
    pred= clf.predict(data)
    return render_template('after.html', data=pred)
        

if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)
    
    
     
        
        
    
    
    
     
        
        

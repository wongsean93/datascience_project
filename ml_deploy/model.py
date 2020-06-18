# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:45:53 2020

@author: Sean
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import pickle


os.chdir('C:\\Users\\Sean\\Desktop\\ml_deploy')

df = pd.read_csv("clean_df_vader.csv")

vader_score = []

for i in df.sentiment_vader:
    if i <= 0:
        score = 0
    else:
        score = 1
    vader_score.append(score)
df["vader_score"] = vader_score

svm_data = pd.DataFrame({"final_clean" : df["final_clean"], "vader_score" : df["vader_score"]})
train, test = train_test_split(svm_data, test_size=0.3, random_state=101)

vader_train_1 = train[train['vader_score']==0]
vader_train_2 = train[train['vader_score']==1]

vader_train_1 = train[train['vader_score']==0].sample(45000,replace=True, random_state=53)
vader_train_2 = train[train['vader_score']==1].sample(45000,replace=True, random_state=53)
vader_training_bs = pd.concat([vader_train_1, vader_train_2])

train['vader_score'].value_counts(normalize=True)
vader_training_bs['vader_score'].value_counts(normalize=True)

Train_X = vader_training_bs["final_clean"]
Train_Y = vader_training_bs["vader_score"]
Test_X = test["final_clean"]
Test_Y = test["vader_score"]
Encoder = LabelEncoder()


SVM = LinearSVC(random_state=0, loss="hinge")
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
sv_params = {'C':[1], 'intercept_scaling':[1]}

tfvec_b = TfidfVectorizer(ngram_range=(1,3))
X_train_cvec = tfvec_b.fit_transform(Train_X.astype(str))

grid_svm = GridSearchCV(SVM, param_grid=sv_params, n_jobs=-1, verbose=1) 


X_test_cvec = tfvec_b.transform(Test_X.astype(str))

grid_svm.fit(X_train_cvec, Train_Y)


pickle.dump(tfvec_b, open('tfidf','wb'))
pickle.dump(grid_svm, open('svm_model.pkl','wb'))
model = pickle.load(open('svm_model.pkl', 'rb'))  






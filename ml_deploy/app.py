# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

os.chdir('C:\\Users\\Sean\\Desktop\\ml_deploy')

app = Flask(__name__)
model = pickle.load(open('svm_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = pd.Series(int_features)
    rev_train_cvec = tfidf.transform(final_features.astype(str))
    prediction = model.predict(rev_train_cvec)

    if prediction == 0:
        output = "{Negative}"
    else:
        output = "{Positive}"
        
    return render_template('index.html', prediction_text='The sentiment class is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
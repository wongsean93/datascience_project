# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:40:33 2020

@author: Sean
"""

import requests


url = 'http://localhost:5000/predict_api'
head = {'comment': '2'}
r = requests.post(url, json=head)


print(r.text)

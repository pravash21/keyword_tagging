from flask import Flask,, request, jsonify
app = Flask(__name__)

import pandas as pd
import json
import numpy as np
from collections import defaultdict
from pandas.io.json import json_normalize
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import json
import os
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from nltk.tag.sequential import ClassifierBasedPOSTagger
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
from bs4 import BeautifulSoup
import pickle
from collections import defaultdict
from collections import Counter
from pandas.io.json import json_normalize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
stem = PorterStemmer()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
@app.route('/api',methods=['POST'])
def tag_pred():
	stop_words = set(stopwords.words("english"))
	new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","how","where","why","it","he","she","of"]
	stop_words = stop_words.union(new_words)

	data = request.get_json(force=True)
	corpus =[]
	soup = BeautifulSoup(data)
	txt = re.sub('[^a-zA-Z]', ' ', soup.get_text())
	txt = txt.lower()
	txt=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",txt)
	txt=re.sub("(\\d|\\W)+"," ",txt)
	txt = txt.split()
	ps=PorterStemmer()
	lem = WordNetLemmatizer()
	txt = [lem.lemmatize(word) for word in txt if not word in stop_words] 
	txt = " ".join(txt)
	corpus.append(txt)
	if(len(corpus[0])!=0):
		cv=CountVectorizer(max_df=1,min_df=0.95,stop_words=stop_words, max_features=10000, ngram_range=(1,2))
		X=cv.fit_transform(corpus)
		keyword = list(cv.vocabulary_.keys())
	else:
		keyword=['']
	max_words = 1000
	tokenize = text.Tokenizer(num_words=max_words, char_level=False)
	mdl = pickle.load(open('mdl.pkl','rb'))
	model_enc = pickle.load(open('enco_model.pkl','rb'))
	xtes = tokenize.texts_to_matrix(keyword)
	k = model_enc.classes_
	arr = []
	for i in xtes:
		pred = mdl.predict(np.array([i]))
		arr.append(k[np.argmax(pred)])
	t_list = set(arr)
	arr = list(t_list)
	test_list = Counter(arr) 
	res = test_list.most_common(1)[0][0]

	return jsonify({'tags': res})
	
	


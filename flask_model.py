# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 255

# instantiate flask 
app = flask.Flask(__name__)

# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()
model = load_model('rnn_model.h5')

import re
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts

import pickle
with open("tokenizer.pkl", "rb") as input_file:
    tokenizer = pickle.load(input_file)

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        cmt = params.get("msg")
        cmt = normalize_texts([cmt])
        cmt = tokenizer.texts_to_sequences(cmt)
        cmt = pad_sequences(cmt, maxlen=MAX_LENGTH)
        with graph.as_default():
            data["prediction"] = str(model.predict(cmt)[0][0])
            data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(host='0.0.0.0')
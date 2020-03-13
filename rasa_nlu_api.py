# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:12:18 2020

@author: mwahdan
"""

from flask import Flask, jsonify, request
from config import DevConfig
import argparse
import os
from rasa_nlu.model import Interpreter
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.models import load_model
import jieba
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import model.modelconfig as path
import model.qa_inference as qa_inference
import json


interpreter_mba = Interpreter.load(path.model['mb'])
interpreter_proj = Interpreter.load(path.model['proj'])
interpreter_bill = Interpreter.load(path.model['bill'])

sess = tf.Session()
set_session(sess)
graph = tf.get_default_graph()
w2vmodel = Word2Vec.load(path.model['w2v'])
lstmmodel = load_model(path.model['lstm'])
label_dic = {'合約查詢': 0, '帳務查詢': 1, '魔速方塊': 2}
gensim_dict = Dictionary()
gensim_dict.doc2bow(w2vmodel.wv.vocab.keys(), allow_update=True)
w2id = {v: k + 1 for k, v in gensim_dict.items()}

qa_inference.init_inference_Engine()

# Create app
app = Flask(__name__)
app.config.from_object(DevConfig)

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello from NLU service'

@app.route('/clf', methods=['GET', 'POST'])
def clf():
    data = request.get_json()
    new_sen_list = jieba.lcut(data['text'])
    sen2id = [w2id.get(word,0) for word in new_sen_list]
    sen_input = pad_sequences([sen2id], maxlen=200)
    global lstmmodel
    global sess
    with graph.as_default():
        set_session(sess)
        res = lstmmodel.predict(sen_input)[0]

    probability = res[np.argmax(res)]
    result = list(label_dic.keys())[list(label_dic.values()).index(np.argmax(res))]

    return jsonify({"probability": str(probability), "result": result})


@app.route('/mb', methods=['GET', 'POST'])
def mb():
    data = request.get_json()
    result_int = interpreter_mba.parse(data['text'])

    return jsonify(result_int)

@app.route('/proj', methods=['GET', 'POST'])
def proj():
    data = request.get_json()
    result_int = interpreter_proj.parse(data['text'])

    return jsonify(result_int)

@app.route('/bill', methods=['GET', 'POST'])
def bill():
    data = request.get_json()
    result_int = interpreter_bill.parse(data['text'])

    return jsonify(result_int)

@app.route('/mbqa', methods=['GET', 'POST'])
def mbqa():
    data = request.get_json()
    my_result = qa_inference.fast_do_inference( _qas_id = 0, _question_text = data['text'], _doc_tokens = data['qa_text'])
    
    return jsonify(json.loads(json.dumps(my_result)))


if __name__ == '__main__':
    
   
    print(('Starting the Server'))
    # Run app
    app.run(host='0.0.0.0', port=40000)

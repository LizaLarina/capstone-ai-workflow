import numpy as np
import pandas as pd
import argparse
from flask import request, Flask, jsonify
from flask import render_template, send_from_directory
import os
import re
import socket
import json
import joblib
from model import model_load, model_train, model_predict
from model import MODEL_VERSION, MODEL_VERSION_NOTE

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/running', methods=['POST'])
def running():
    return render_template('running.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if not request.json:
        print("No request data")
        return jsonify([])
    if 'country' not in request.json:
        print("Please provide the country name")
        return jsonify([])
    if 'day' not in request.json:
        print("Please provide the day")
        return jsonify([])
    if 'month' not in request.json:
        print("Please provide the month")
        return jsonify([])
    if 'year' not in request.json:
        print("Please provide the year")
        return jsonify([])
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True
    country = request.json['country']
    day = request.json['day']
    month = request.json['month']
    year = request.json['year']
    data_dir = os.path.join(".","data","cs-train")
    all_data, all_models = model_load(data_dir=data_dir)
    model = all_models[country]
    if not model:
        print("Mo models avaliable")
        return jsonify([])
    _result = model_predict(country, year, month, day,test=test)
    result = {}
    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item
    return(jsonify(result))

@app.route('/train', methods=['GET','POST'])
def train():
    if not request.json:
        print("No request data")
        return jsonify(False)
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True
    print("Training in progress..")
    data_dir = os.path.join(".","data","cs-train")
    print(data_dir)
    model = model_train(data_dir, test=test)
    print("Training is completed!")
    return(jsonify(True))

@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    if not re.search(".log",filename):
        print("It is not a log file: {}".format(filename))
        return jsonify([])
    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("No such directory")
        return jsonify([])
    file_path = os.path.join(log_dir, filename)
    if not os.path.exists(file_path):
        print("No file found {}".format(filename))
        return jsonify([])
    return send_from_directory(log_dir, filename, as_attachment=True)

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())
    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True, port=8080)

import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    data_2d = np.array(list(data.values())).reshape(1,-1)
    scaled_data_2d = scaler.transform(data_2d)
    pred = model.predict(scaled_data_2d)[0]
    output = np.round(pred,2)
    return jsonify(output)


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data_2d = np.array(data).reshape(1,-1)
    scaled_data_2d = scaler.transform(data_2d)
    pred = model.predict(scaled_data_2d)[0]
    output = np.round(pred,2)
    return render_template('home.html', prediction_text = "The House Price Prediction is {}".format(output))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



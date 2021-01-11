from flask import Flask
from flask import request
import numpy as np
import pandas as pd
import pickle
import json
import os
# import sklearn
app = Flask(__name__)

with open('tree_model.pickle', 'rb') as f:
    tree_cls = pickle.load(f)

keys = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width']


@app.route('/')
def index():
    return 'Welcome to The Iris prediction demo'


@app.route('/iris/predict_single')
def predict_single():
    if any(key not in request.args for key in keys):
        return "Bad Request", 400

    sepal_len = request.args.get('sepal_len')
    sepal_width = request.args.get('sepal_width')
    petal_len = request.args.get('petal_len')
    petal_width = request.args.get('petal_width')
    X = np.array([sepal_len, sepal_width, petal_len, petal_width]).reshape(1, -1)
    y_pred = tree_cls.predict(X)
    return str(y_pred[0])


@app.route('/iris/predict_many', methods=["POST"])
def predict_many():
    if not request.is_json:
        return "Not a Valid Request", 400

    X = pd.DataFrame(json.loads(request.get_json())).to_numpy()
    y_pred = tree_cls.predict(X)
    return json.dumps(list(y_pred))


if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()

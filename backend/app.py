from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.DecisionTree import DTClassifier
from utils.LinearRegression import LinearRegression
app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route("/DT", methods=['POST'])
def dt():
    file_train = request.files.get("train")
    file_test = request.files.get("test")
    file_train.save('./src/' + "train.csv")
    file_test.save('./src/' + "test.csv")
    params = request.form
    LinearRegression(params)
    return jsonify({
        'msg': "成功收到文件"
    })

if __name__ == '__main__':
    app.run(port=12000)

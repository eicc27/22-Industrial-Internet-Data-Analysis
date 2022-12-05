from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.DecisionTree import DecisionTreeClassifier
from utils.LinearRegression import LinearRegression
from utils.SVMClassifier import SVMClassifier
from flask import send_from_directory
app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route("/linearRegression", methods=['POST'])
def linearReg():
    file_train = request.files.get("train")
    file_test = request.files.get("test")
    file_train.save('./src/' + "train.csv")
    file_test.save('./src/' + "test.csv")
    params = request.form
    LinearRegression(params)
    return jsonify({
        'code':200,
        'msg': "成功收到文件"
    })


@app.route("/decisionTreeClassfier", methods=['POST'])
def decisionTree():
    file_train = request.files.get("train")
    file_test = request.files.get("test")
    file_train.save('./src/' + "train.csv")
    file_test.save('./src/' + "test.csv")
    params = request.form
    DecisionTreeClassifier(params)
    return jsonify({
        'msg': "成功收到文件"
    })


@app.route("/SVMClassifier", methods=['POST'])
def SVM():
    file_train = request.files.get("train")
    file_test = request.files.get("test")
    file_train.save('./src/' + "train.csv")
    file_test.save('./src/' + "test.csv")
    params = request.form
    SVMClassifier(params)
    return jsonify({
        'msg': "成功收到文件"
    })


@app.route("/getResult", methods=['POST'])
def getResult():
    return send_from_directory('./src', 'test_result.csv')


@app.route("/getModel", methods=['POST'])
def getModel():
    return send_from_directory('./src', 'test_result.csv')


if __name__ == '__main__':
    DecisionTreeClassifier(111)
    app.run(port=12000)

from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.DecisionTree import DecisionTreeClassifier, DecisionTreeClassifierPredict
from utils.LinearRegression import LinearRegression, LinearRegressionPredict
from utils.SVMClassifier import SVMClassifier, SVMClassifierPredict
from flask import send_from_directory
app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route("/linearRegression", methods=['POST'])
def linearReg():
    try:
        file_train = request.files.get("train")
        file_test = request.files.get("test")
        file_train.save('./src/' + "train.csv")
        file_test.save('./src/' + "test.csv")
    except:
        return jsonify({
            'code': 400,
            'msg': "文件接收失败"
        })

    try:
        loss = request.form.get('loss')
        max_iter = int(request.form.get('max_iter'))
        shuffle = request.form.get('shuffle') == 'true'
        tol = float(request.form.get('tol'))

    except:
        return jsonify({
            'code': 401,
            'msg': "参数接收失败"
        })

    try:
        error = LinearRegression(loss, max_iter, shuffle, tol)
        return jsonify({
            'code': 200,
            'msg': "训练成功",
            'error' : error
        })
    except:
        return jsonify({
            'code': 402,
            'msg': "训练失败"
        })



@app.route("/decisionTreeClassfier", methods=['POST'])
def decisionTree():
    try:
        file_train = request.files.get("train")
        file_test = request.files.get("test")
        file_train.save('./src/' + "train.csv")
        file_test.save('./src/' + "test.csv")
    except:
        return jsonify({
            'code': 400,
            'msg': "文件接收失败"
        })

    try:
        criterion = request.form.get('criterion')
        max_depth = int(request.form.get("max_depth"))
        max_leaf_nodes = int(request.form.get("max_leaf_nodes"))

    except:
        return jsonify({
            'code': 401,
            'msg': "参数接收失败"
        })

    try:
        accuracy = DecisionTreeClassifier(criterion, max_depth, max_leaf_nodes)
        return jsonify({
            'code': 200,
            'msg': "训练成功",
            'accuracy': accuracy
        })
    except:
        return jsonify({
            'code': 402,
            'msg': "训练失败"
        })

@app.route("/SVMClassifier", methods=['POST'])
def SVM():
    try:
        file_train = request.files.get("train")
        file_test = request.files.get("test")
        file_train.save('./src/' + "train.csv")
        file_test.save('./src/' + "test.csv")
    except:
        return jsonify({
            'code': 400,
            'msg': "文件接收失败"
        })

    try:
        C = float(request.form.get('C'))
        kernel = request.form.get("kernel")
        tol = float(request.form.get('tol'))
        max_iter = int(request.form.get("max_iter"))

    except:
        return jsonify({
            'code': 401,
            'msg': "参数接收失败"
        })

    try:
        accuracy = SVMClassifier(C, kernel, tol, max_iter)
        return jsonify({
            'code': 200,
            'msg': "训练成功",
            'accuracy': accuracy
        })
    except:
        return jsonify({
            'code': 402,
            'msg': "训练失败"
        })


@app.route("/linearRegressionPredict", methods=['POST'])
def linearRegPredict():
    try:
        file_predict = request.files.get("predict")
        file_model = request.files.get("model")
        file_predict.save('./src/' + "predict.csv")
        file_model.save('./src/' + "model.model")
        LinearRegressionPredict()
        return send_from_directory("./src", 'result.csv')
    except:
        return jsonify({
            'code': 400,
            'msg': "预测失败"
        })


@app.route("/DecisionTreeClassifierPredict", methods=['POST'])
def DecisionTreeClassifierPred():
    try:
        file_predict = request.files.get("predict")
        file_model = request.files.get("model")
        file_encoder = request.files.get('encoder')
        file_predict.save('./src/' + "predict.csv")
        file_encoder.save('./src/' + "encoder.model")
        file_model.save('./src/' + "model.model")

        DecisionTreeClassifierPredict()
        return send_from_directory("./src", 'result.csv')
    except:
        return jsonify({
            'code': 400,
            'msg': "预测失败"
        })


@app.route("/SVMClassifierPredict", methods=['POST'])
def SVMClassifierPred():
    try:
        file_predict = request.files.get("predict")
        file_model = request.files.get("model")
        file_encoder = request.files.get('encoder')
        file_predict.save('./src/' + "predict.csv")
        file_model.save('./src/' + "model.model")
        file_encoder.save('./src/' + "encoder.model")
        SVMClassifierPredict()
        return send_from_directory("./src", 'result.csv')
    except:
        return jsonify({
            'code': 400,
            'msg': "预测失败"
        })


@app.route("/getModel", methods=['POST'])
def getModel():
    return send_from_directory('./src', 'model.model')


@app.route("/getEncoderModel", methods=['POST'])
def getEncoderModel():
    return send_from_directory('./src', 'encoder.model')


# @app.route("/getResult", methods=['POST'])
# def getResult():
#     return send_from_directory('./src', 'result.csv')



if __name__ == '__main__':
    app.run(debug=True, port=12000)

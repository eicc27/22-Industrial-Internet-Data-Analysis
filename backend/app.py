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
        random_state = int(request.form.get('random_state'))
        print(random_state)
        tol = float(request.form.get('tol'))
        print(tol)
        penalty = request.form.get('penalty')
        alpha = float(request.form.get('alpha'))
        print(loss, max_iter, shuffle, random_state, tol, penalty, alpha)
    except:
        return jsonify({
            'code': 401,
            'msg': "参数接收失败"
        })

    try:
        error = LinearRegression(loss, max_iter, shuffle, random_state, tol, penalty, alpha)
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
        random_state = int(request.form.get('random_state'))
        splitter = request.form.get("splitter")
        max_depth = int(request.form.get("max_depth"))
        max_leaf_nodes = int(request.form.get("max_leaf_nodes"))
        min_samples_leaf = int(request.form.get("min_samples_leaf"))
        min_samples_split = int(request.form.get("min_samples_split"))
        print(criterion, random_state, splitter, max_depth, max_leaf_nodes, min_samples_split, min_samples_leaf)
    except:
        return jsonify({
            'code': 401,
            'msg': "参数接收失败"
        })

    try:
        accuracy = DecisionTreeClassifier(criterion, random_state, splitter, max_depth, max_leaf_nodes, min_samples_split, min_samples_leaf)
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
        degree = int(request.form.get("degree"))
        tol = float(request.form.get('tol'))
        max_iter = int(request.form.get("max_iter"))
        random_state = int(request.form.get('random_state'))
        print(C, kernel, degree, tol, max_iter, random_state)
    except:
        return jsonify({
            'code': 401,
            'msg': "参数接收失败"
        })

    try:
        accuracy = SVMClassifier(C, kernel, degree, tol, max_iter, random_state)
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

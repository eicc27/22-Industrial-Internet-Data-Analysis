from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.DecisionTree import DecisionTreeClassifier, DecisionTreeClassifierPredict
from utils.LinearRegression import LinearRegression, LinearRegressionPredict
from utils.SVMClassifier import SVMClassifier, SVMClassifierPredict
from flask import send_from_directory
import numpy as np
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.abspath('.'))

from preproc.feature_selection import FeatureSelection
from preproc.sifting import Sifting
from preproc.normalization import Norm
from preproc.padding import Padding
from preproc.dataloader import Dataloader
from preproc.utilities import Utils
from preproc.logger import Logger


app = Flask(__name__)
CORS(app, resources=r'/*')


@app.route("/preproc_intro", methods=['GET'])
def intro():
    return jsonify(json.load(open('./intro.json', 'r')))


@app.route("/upload", methods=['post'])
def upload():
    try:

        file = request.files.get("file")
        print(file.name)
        file.save("./src/data.csv")
        print(file.filename)
        # Utils.strs_to_csv("./src/data.csv")
    except:
        return jsonify({
            'code': 400,
            'msg': "文件接收失败"
        })
    return jsonify({
        'code': 200,
        'msg': "文件上传成功"
    })


@app.route("/preproc", methods=['post'])
def preproc():
    try:
        data = Dataloader('./src/data.csv').load()
        labels = data.columns.values
        data = data.to_numpy()
    except ValueError:
        return jsonify({
            'code': 400,
            'msg': "文件读取失败"
        })
    req: dict = json.loads(request.get_data(as_text=True))
    pred = data[:, [req['pred_column']]]
    data_p = data[:, req['data_columns']]
    data = np.concatenate([data_p, pred], axis=1)
    labels_p = []
    for c in req['data_columns']:
        labels_p.append(labels[c])
    labels_p.append(labels[req['pred_column']])
    if req['padding']:
        for i, padding in enumerate(req['padding']):
            if padding and len(padding):
                # try:
                    Logger(f'Padding column {labels_p[i]} using {padding}').log('info')
                    data = Padding(data, i, padding).run()
                # except ValueError:
                #     return jsonify({
                #         'code': 400,
                #         'msg': "padding算法出错",
                #         'column_index': i
                #     })
    pd.DataFrame(data).to_csv('./d.csv')
    if req['norm']:
        for i, norm in enumerate(req['norm']):
            if norm:
                try:
                    data = Norm(data, i, norm).run()
                except ValueError:
                    return jsonify({
                        'code': 400,
                        'msg': "norm算法出错",
                        'column_index': i
                    })
    if req['sifting']:
        try:
            print(req['sifting']['method'])
            data = Sifting(data, req['sifting']['method'], req['pred_column'], req['sifting']['th']).run()
        except ValueError as e:
                    print(e)
                    return jsonify({
                        'code': 400,
                        'msg': "sift算法出错"
                    })
    df = pd.DataFrame(data, columns=labels_p)
    df.to_csv('./src/data_preproc.csv', index=False)
    return send_from_directory('./src', 'data_preproc.csv')

@app.route('/fs', methods=['POST'])
def feature_selection():
    req: dict = json.loads(request.get_data(as_text=True))
    dl = Dataloader('./src/data_preproc.csv').load()
    data = dl.to_numpy()
    labels = dl.columns.values
    pred = data[:, [req['pred_column']]]
    data_p = data[:, req['data_columns']]
    data = np.concatenate([data_p, pred], axis=1)
    labels_p = []
    for c in req['data_columns']:
        labels_p.append(labels[c])
    try:
        FeatureSelection(data[:, :-1], labels_p, data[:, -1], req['method']).savefig()
    except ValueError:
        return jsonify({
                        'code': 400,
                        'msg': "fs算法出错",
                    })
    return send_from_directory('./src', 'fs.png')



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
            'error': error
        })
    except:
        return jsonify({
            'code': 402,
            'msg': "训练失败"
        })


@app.route("/decisionTreeClassifier", methods=['POST'])
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
        return jsonify({
            'code': 200,
            'msg': "预测成功"
        })
    except:
        return jsonify({
            'code': 400,
            'msg': "预测失败"
        })


@app.route("/decisionTreeClassifierPredict", methods=['POST'])
def DecisionTreeClassifierPred():
    try:
        file_predict = request.files.get("predict")
        file_model = request.files.get("model")
        file_encoder = request.files.get('encoder')
        file_predict.save('./src/' + "predict.csv")
        file_encoder.save('./src/' + "encoder.model")
        file_model.save('./src/' + "model.model")

        DecisionTreeClassifierPredict()
        return jsonify({
            'code': 200,
            'msg': "预测成功"
        })
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
        return jsonify({
            'code': 200,
            'msg': "预测成功"
        })
    except:
        return jsonify({
            'code': 400,
            'msg': "预测失败"
        })


@app.route("/getModel", methods=['get'])
def getModel():
    return send_from_directory('./src', 'model.model')


@app.route("/getEncoderModel", methods=['get'])
def getEncoderModel():
    return send_from_directory('./src', 'encoder.model')


@app.route("/getResult", methods=['POST'])
def getResult():
     return send_from_directory('./src', 'result.csv')


if __name__ == '__main__':
    app.run(debug=True, port=12000)

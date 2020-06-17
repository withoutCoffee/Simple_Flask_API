import numpy as np
from flask import Flask, jsonify, request
from flask_restful  import Api, Resource
import pickle

# Carregando o modelo para o código
with open("./utils/model.pickle","rb") as f:
    # Carrega o arquico model.pickle em modeo read binary
    modelo_carregado = pickle.load(f)

app = Flask(__name__)

# Inicia o swagger app
rest_app = Api(app)

class HelloWorld(Resource):
    def post(self):
        try:
            array = request.json['array']
            # Usando o algoritmo de M.L
            array = np.array(array).reshape(-1,1)
            pred = modelo_carregado.predict(np.array(array))

            return{
                "status": "Array recebido",
                "Quantidade_de_numeros_recebidos_para_prever:": array.shape[0],
                "valores_requisicao" : array.T[0].tolist(),
                "valores_preditos": pred.T[0].tolist(),
            },200
        except Exception as e:
            return {"Error":"Parâmetros incorretos"},400

    def get(self):
        return{"status":"[1,2,3]"}

rest_app.add_resource(HelloWorld, '/')

if __name__ == "__main__":
    # Com o uso do degub = true quando o arquivo for atualizado o o server dar restart
    debug = True
    app.run(host = '127.0.0.1',port=5000,debug = debug)


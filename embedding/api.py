from flask import Flask, request
from flask_restful import Resource, Api
from model import EmbeddingModel

app = Flask(__name__)
api = Api(app)

model = EmbeddingModel()

class EmbeddingApi(Resource):
    
    def get(self):
        return (model.encode(request.form['data']))

api.add_resource(EmbeddingApi, '/')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify, Response 
from flask_restful import Resource, Api
import pandas as pd
import pickle
import preprocessing as pre

model = pickle.load(open('model.pkl', 'rb'))
corpus = pickle.load(open('corpus.pkl', 'rb'))

app = Flask(__name__)
api = Api(app)


class status(Resource):
	def get(self):
		return jsonify({"Status": 'Api is running'})

class predict(Resource):
	def post(self):
		text_dict = request.get_json(force=True)
		text_df = pd.DataFrame([text_dict['text']], columns = ['text'], index = [0])
		pre.preprocess_text_cols(text_df)
		pre.create_new_features(text_df)

		global corpus
		word_features_train_df = pre.tf_idf(text_df, 'final_text', corpus)
		train, features = pre.return_final_df(text_df, word_features_train_df, target_series=None)

		prediction = int(model.predict(train[features])[0])

		return jsonify({"result": prediction})


# Endpoints

api.add_resource(status, "/status")
api.add_resource(predict, "/predict.json")

# Run app

if __name__ == "__main__":
	app.run(debug=False, port=3000)
from flask import Flask, request, jsonify, Response, render_template, make_response 
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
	def get(self):
		headers = {'Content-Type': 'text/html'}
		return make_response(render_template('base_form.html'),200,headers)


	def post(self):
		text_dict = request.form.to_dict()
		text_df = pd.DataFrame([text_dict['text']], columns = ['text'], index = [0])
		pre.preprocess_text_cols(text_df)
		pre.create_new_features(text_df)

		global corpus
		word_features_train_df = pre.tf_idf(text_df, 'final_text', corpus)
		train, features = pre.return_final_df(text_df, word_features_train_df, target_series=None)

		prediction = int(model.predict(train[features])[0])
		headers = {'Content-Type': 'text/html'}

		if prediction == 0:
			return make_response(render_template('prediction_form.html', prediction = prediction, text = text_dict['text']), 200, headers)
		elif prediction == 1:
			return make_response(render_template('prediction_form.html', prediction = prediction, text = text_dict['text']), 200, headers)
		else:
			return make_response(render_template('prediction_form.html', prediction = 'Error', text = text_dict['text']), 200, headers)


# Endpoints

api.add_resource(status, "/status")
api.add_resource(predict, "/predict")

# Run app

if __name__ == "__main__":
	app.run(debug=True, port=3000)
import requests
import json
import unittest


class TestStatus(unittest.TestCase):

	def test_status(self):
		port = 3000
		url = 'http://localhost:' + str(port) + '/status'
		response = requests.get(url)

		self.assertEqual(response.json()['Status'], "Api is running", "Should be: 'Api is running'")
		self.assertEqual(response.status_code, 200, "Should be: 200")


class TestPredict(unittest.TestCase):

	def test_predict(self):
		port = 3000
		url = 'http://localhost:' + str(port) + '/predict.json'

		test_tweet = 'This is a test'
		tweet_json = json.dumps({"text": test_tweet})

		response = requests.post(url, tweet_json)

		self.assertEqual(response.json()['result'], 0, "Should be: 0")
		self.assertEqual(response.status_code, 200, "Should be: 200")

	def test_blank_predict(self):
		port = 3000
		url = 'http://localhost:' + str(port) + '/predict.json'

		test_tweet = ''
		tweet_json = json.dumps({"text": test_tweet})

		response = requests.post(url, tweet_json)

		self.assertEqual(response.json()['result'], 'Error', "Should be: 0")
		self.assertEqual(response.status_code, 200, "Should be: 200")


if __name__ == '__main__':
	unittest.main()
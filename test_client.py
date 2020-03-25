import requests
import json

#port = 3000
#url = 'http://localhost:' + str(port) + '/predict.json'

url = 'https://tweet-disaster-prediction.herokuapp.com/predict.json'

test_tweet = 'This is a test'
tweet_json = json.dumps({"text": test_tweet})

send_request = requests.post(url, tweet_json)

print(send_request)
print(send_request.json())
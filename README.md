# Tweet Disaster Prediction API
A Machine Learning API that utilizes Natural Language Processing (NLP) for predicting whether a tweet is about an actual disaster or not.

## Technologies/Libraries Used
- Python
- Flask
- SK Learn
- Heroku

## Running API Locally

In order to run this program locally, create a new python virtual environment using python 3.7 and launch the API by following the steps listed below:

1. Download and install the newest verion of python from the python downloads page ([here](https://www.python.org/downloads/)).
2. Run `python3 -m venv new_env` from the command line to create a new virtual environment in your current directory. 
3. Activate the virtual environment by running  `source new_env/bin/activate`,
4. Clone this repo using `git clone` or download the repository and unzip the files to this location.
5. Install the required packages using `pip install -r requirements.txt`.
6. Launch the API using `python app.py`

## Tests

Once the API is running, test the API by following the steps below:
1. Open another terminal window/tab.
2. Activate the virtual environment using `source path/to/new_env/new_env/bin/activate`.
3. Test to make sure the API is running using `curl http://localhost:3000/status`. This should return `{"Status": 'Api is running'}`. 
4. Test to ensure the API is working correctly using the tests provided by running `python -m unittest -v test/test_local_client.py`.   


## Interacting with the API - Production and Local Versions
```
{bash}
# Getting a prediction
$ curl -i -X POST -d '{"text": "This is a test"}' http://localhost:3000/predict.json
HTTP/1.0 200 OK
...
{
"result": 0
}
```

A result of 1 indicates the model predicts the tweet is about an actual disaster while a result of 0 indicates that the model predicts the tweet is not about an actual disaster.

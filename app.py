# Import libraries
from flask import Flask, render_template,request
import pandas as pd
import yaml
from sklearn import preprocessing
from tensorflow.keras.models import load_model
import flask


# Initialize the flask App
app = Flask(__name__, template_folder = "templates")

# Define the Path to the Config File
path_to_config_file = "config.yaml"

def config_params():

    with open(path_to_config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    data_path = cfg["water_consumption"]["preprocessed_data"]
    model_path = cfg["water_consumption"]["model_path"]
    n_past_cfg = cfg["Settings"]["n_past"]
    n_features_cfg = cfg["features"]["n_features"]

    return data_path, model_path, n_past_cfg, n_features_cfg


# Function to get Last 48 hours data for all households and Predict for the next hour
def make_predictions():
    
    data_path, model_path, n_past_cfg, n_features_cfg = config_params()

    # Read the Data (And get the Last 48 Hours Data)
    df = pd.read_csv(data_path)
    df_test = df.tail(n = n_past_cfg)

    df_test = df_test.rename({"Unnamed: 0":"time"}, axis = 1)
    df_test = df_test.set_index('time')

    # Scaling the Data
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df_test)
    data = data.reshape(1, n_past_cfg, n_features_cfg)

    # Load the model and make the prediction
    model = load_model(model_path)
    prediction = model.predict(data)
    prediction = prediction.flatten()
    prediction = prediction.reshape(1, n_features_cfg)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction.flatten().tolist()

    # View and Check the predicted values
    new_prediction = []
    for i in prediction:
        if float(i) < 0:
            new_prediction.append(0)
        else:
            try:
                new_prediction.append(float(round(i,1)))
            except:
                new_prediction.append(float(i))

    return new_prediction


# Default page of the web-app
@app.route('/', methods=['GET', 'POST'])


def main():

    if flask.request.method == 'GET':
        return render_template('index.html')

    if flask.request.method == 'POST':
        search_string = flask.request.form['search_string']

        if search_string.lower() == "yes":
            predictions = make_predictions()

        return render_template('index.html', original_input={'search_string':search_string}, result=predictions)


# To use the Submit button in the web-app
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

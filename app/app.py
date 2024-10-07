import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import cloudpickle as cp
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
APP_MODEL_NAME = os.getenv('APP_MODEL_NAME')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

#Load the model from mlflow
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
model_name = APP_MODEL_NAME 
model_version = '1'
loaded_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

# load scaler
scaler_filename = './model/scaler.pkl'
sc = pickle.load(open(scaler_filename, 'rb'))


df = pd.read_csv('./datasets/Cars.csv')
df.rename(columns = {'name':'brand',}, inplace = True)
df['brand'] = df['brand'].str.split(' ').str.get(0)
le = LabelEncoder()
enCodeBrandName = df['brand'] = le.fit_transform(df['brand'])
deCodeBrandName = df['brand'] = le.inverse_transform(enCodeBrandName)
years = list(range(1995, 2018))
percentiles = np.percentile(df['selling_price'], [0, 25, 50, 75, 100])
options = []
for label, value in zip(np.unique(deCodeBrandName), np.unique(enCodeBrandName)):
    options.append({'label': label, 'value': value})

#set up webapp
# Initialize the Dash app
app = dash.Dash(__name__)

# Create the layout with dropdowns or input fields
app.layout = html.Div(children=[
    html.H1('Predicting Car Price 3 Chacky Company', style={'textAlign': 'center','color': '#37F420','fontSize': '40px','border': '2px solid black','backgroundColor': 'Green'}),
    html.H2('Instruction', style={'color': 'white'}),
    html.P('Hello this is instruction that how prediction work', style={'color': 'white'}),
    html.P('- You need select just one Brand before you click predict', style={'color': 'white'}),
    html.P('- Please fill all the field that will make this app to be the best accuracy of prediction', style={'color': 'white'}),
    html.P('- For the field you need to fill only integer or float (If string it will be None value except Brand)', style={'color': 'red','fontWeight': 'bold'}),
    html.P('Enjoy Prediction!', style={'color': 'white'}),
    # Dropdown example
    html.Br(),
    html.Label('Select Brand:', style={'color': 'white'}),
    dcc.Dropdown(
        id='feature-dropdown',
        options=options,
        value= 20  # Default value
    ),
    html.Br(),
    html.Label('Enter a Year from 1995 to 2017 :', style={'color': 'white'}),
     dcc.Dropdown(
        id='feature-Year',
        options=[{'label': str(year), 'value': year} for year in years],
        value=None  # Default value
    ),
    html.Br(),
    html.Label('Mileage:', style={'color': 'white'}),
    dcc.Input(id='feature-Mileage', type='number', value=np.nan,style={'marginRight': '20px','width': '150px','height': '30px','borderRadius': '10px'}),

   
    html.Label('KM-driven:', style={'color': 'white'}),
    dcc.Input(id='feature-kmdriven', type='number', value=np.nan,style={'marginRight': '20px','width': '150px','height': '30px','borderRadius': '10px'}),

   
    html.Label('Engine:', style={'color': 'white'}),
    dcc.Input(id='feature-engine', type='number', value=np.nan,style={'marginRight': '20px','width': '150px','height': '30px','borderRadius': '10px'}),

    
    html.Label('Max-power:', style={'color': 'white'}),
    dcc.Input(id='feature-max_power', type='number', value=np.nan,style={'marginRight': '20px','width': '150px','height': '30px','borderRadius': '10px'}),

    
    html.Br(),
    html.Br(),
    html.P('- the value prediction will show the range of price', style={'color': 'red','fontWeight': 'bold'}),
    html.Button(
        'Predict', 
        id='predict-button',
        style={
            'backgroundColor': 'Green',  # Button background color
            'color': 'white',           # Text color
            'fontSize': '20px',         # Font size of the button text
            'padding': '10px 20px',     # Padding inside the button
            'borderRadius': '10px',     # Rounded corners
            'border': '2px solid black' # Border style
        }, n_clicks=0
    ),
    #html.Button('Predict', id='predict-button'),

    # Output container for the prediction
    html.Div(id='prediction-output', style={'margin-top': '50px', 'font-size': '50px', 'textAlign': 'center','color':'red'}),
],style={
        'backgroundColor': '#2E86C1',  # Set the background color for the entire Div
        'height': '100vh',  # Optional: Make the Div cover the full viewport height
        'padding': '20px'   # Optional: Add some padding
    })
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('feature-dropdown', 'value'), Input('feature-Year', 'value'),
     Input('feature-Mileage', 'value'),Input('feature-kmdriven', 'value'),
     Input('feature-engine', 'value'),Input('feature-max_power', 'value')]
)
def update_prediction(n_clicks, dropdown_value,year,mileage,km_driven,engine,max_power):
    if n_clicks and n_clicks > 0:
        if dropdown_value != None:
            if mileage == None:
                mileage = 0
            if km_driven == None:
                km_driven = 0
            if engine == None:
                engine = 0
            if max_power == None:
                max_power = 0

            X_input = np.array([[dropdown_value, mileage, year, km_driven, engine, max_power]])
            print(X_input)

            sample_scaled = sc.transform(X_input)

            # Add intercept term to match the training data structure
            intercept = np.ones((sample_scaled.shape[0], 1))  # Add a column of ones (bias term)
            sample_scaled_with_intercept = np.concatenate([intercept, sample_scaled], axis=1)

            prediction = loaded_model.predict(sample_scaled_with_intercept)

            price = (percentiles[prediction[0]],percentiles[prediction[0]+1])
            resultPredict = f'The Car Price Prediction around {price[0]:.2f} to {price[1]:.2f}'
            print(f"result of prediction : {resultPredict}")
            return resultPredict
        else:
            return "Please Choose at least one"
    return ''


if __name__ == '__main__':
    app.run(debug=True)
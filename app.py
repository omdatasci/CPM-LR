import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from flask import Flask,render_template,request
app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = {
        'Material_Quantity_gm': float(request.form['Material_Quantity']),
        'Additive_Catalyst_gm': float(request.form['Additive_Catalyst']),
        'Ash_Component_gm': float(request.form['Ash_Component']),
        'Water_Mix_ml': float(request.form['Water_Mix']),
        'Plasticizer_gm': float(request.form['Plasticizer']),
        'Moderate_Aggregator': float(request.form['Moderate_Aggregator']),
        'Formulation_Duration_hrs': float(request.form['Formulation_Duration'])
    }
   
    input_df = pd.DataFrame([input_data]) #This line creates a pandas DataFrame named input_df from a dictionary input_data. 
    input_poly = PolynomialFeatures(degree=3).fit_transform(input_df)  #This line applies polynomial feature transformation to the input data. 

    # Make the prediction
    prediction = model.predict(input_poly)[0]

    return render_template('result.html', predictions=prediction)

    


if __name__ == '__main__':
    app.run(debug=True)


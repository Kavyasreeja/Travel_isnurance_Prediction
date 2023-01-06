import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


dir_list = os.listdir(os.getcwd())
for i in dir_list:
    print(i)
    if i.endswith('pkl'):
        picklefile = i
application = Flask(__name__) #Initialize the flask App
model = pickle.load(open(picklefile, 'rb'))

@application.route('/')
def home():
    return render_template('trav.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.values()
    print(features)
    for v in features:
        print(v)
    
    Agency_type = request.form.get("Agency_type")
    Agency=request.form.get("Agency")
    Dist_Channel=request.form.get("Dist_Channel")
    Prod_Name=request.form.get("Prod_Name")
    Duration=request.form.get("Duration")
    Destination=request.form.get("Destination")
    Net_Sales=request.form.get("Net_Sales")
    Commission=request.form.get("Commission")
    Age=request.form.get("Age")
    

    final_features_array = [Agency, Agency_type, Dist_Channel, Prod_Name, Duration, Destination, Net_Sales, Commission, Age]
    # df = pd.DataFrame(final_features)
    # print(final_features)
    # print(df)
    # prediction = model.predict(final_features)
    # output = round(prediction[0], 2)

    # int_features = [int(x) for x in request.form.values()]
    int_features=[]
    for x in final_features_array:
        print(x)
        int_features.append(int(x))
    # int_features = [int(x) for x in final_features_array]
    print(int_features)
    final_features = [np.array(int_features)]
    final_features = pd.DataFrame(final_features)
    prediction = model.predict(final_features)

    output = round(int(prediction[0]), 2)

    if(output==1):
        return render_template('trav.html', prediction_text='CLaimed')
    else:
        return render_template('trav.html', prediction_text='Not CLaimed')


if __name__ == "__main__":
    application.run(debug=True)

from django.shortcuts import render
from django.http import HttpResponse
import pickle
import pandas as pd
import os
import warnings

def index(request):
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    model_path = os.path.join(models_path, 'diabetes.pkl')

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Initialize prediction with None
    prediction = None

    if request.method == 'POST':
        input_data = pd.DataFrame({
            'Pregnancies': [float(request.POST['preg'])],
            'Glucose': [float(request.POST['gul'])],
            'BloodPressure': [float(request.POST['bp'])],
            'SkinThickness': [float(request.POST['st'])],
            'Insulin': [float(request.POST['ins'])],
            'BMI': [float(request.POST['bmi'])],
            'DiabetesPedigreeFunction': [float(request.POST['dpf'])],
            'Age': [float(request.POST['age'])]
        })
        
        # Suppress the warning about feature names
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        prediction = model.predict(input_data)[0]

        print("Prediction:", prediction)

    return render(request, "index.html", {'prediction': prediction})

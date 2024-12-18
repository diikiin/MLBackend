import os

import catboost
import pandas as pd
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATBOOST_MODEL_PATH = os.path.join(BASE_DIR, "catboost_model_best.cbm")
catboost_model = catboost.CatBoostClassifier()
catboost_model.load_model(CATBOOST_MODEL_PATH)

columns_to_scale = ['height', 'weight', 'ap_hi', 'ap_lo', 'age_years', 'bmi']
columns_to_keep = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']


@api_view(['POST'])
def upload_and_predict(request):
    try:
        # Check if a file is uploaded
        if 'file' not in request.FILES:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        uploaded_file = request.FILES['file']
        output_format = request.POST.get("output_format", "csv").lower()  # Default to CSV

        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            return Response({"error": "Unsupported file format. Upload CSV or Excel files."},
                            status=status.HTTP_400_BAD_REQUEST)

        required_columns = ["height", "weight", "ap_hi", "ap_lo", "age_years", "gender",
                           "cholesterol", "gluc", "smoke", "alco", "active"]
        if not all(column in data.columns for column in required_columns):
            return Response({"error": "Missing required columns in the uploaded file"},
                            status=status.HTTP_400_BAD_REQUEST)

        data['bmi'] = (data['weight'] / ((data['height'] / 100) ** 2)).round(2)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
        all_data = pd.concat([scaled_df, data[columns_to_keep].reset_index(drop=True)], axis=1)

        predictions = catboost_model.predict_proba(all_data)[:, 1] * 100
        data["Disease Probability (%)"] = predictions.round(2)

        # Return the processed file
        if output_format == "csv":
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
            data.to_csv(response, index=False)
        elif output_format == "excel":
            response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = 'attachment; filename="predictions.xlsx"'
            data.to_excel(response, index=False, engine='openpyxl')
        else:
            return Response({"error": "Invalid output format. Use 'csv' or 'excel'."},
                            status=status.HTTP_400_BAD_REQUEST)

        return response

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def predict_disease(request):
    try:
        data = request.data

        required_fields = ["height", "weight", "ap_hi", "ap_lo", "age_years", "gender",
                           "cholesterol", "gluc", "smoke", "alco", "active"]
        if not all(field in data for field in required_fields):
            return Response({"error": "Missing required features"}, status=status.HTTP_400_BAD_REQUEST)

        data['bmi'] = (data['weight'] / ((data['height'] / 100) ** 2)).round(2)
        input_data = pd.DataFrame([data])

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(input_data[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
        all_data = pd.concat([scaled_df, input_data[columns_to_keep].reset_index(drop=True)], axis=1)

        # Predict probabilities
        probabilities = catboost_model.predict_proba(all_data)[0]
        prob_disease = probabilities[1] * 100

        return Response({
            "message": "Prediction successful",
            "disease_probability": f"{prob_disease:.2f}%"
        })
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from django.urls import path

from prediction.views import upload_and_predict, predict_disease

urlpatterns = [
    path("predict/upload", upload_and_predict, name="upload_file"),
    path('predict/', predict_disease, name='predict_disease'),
]

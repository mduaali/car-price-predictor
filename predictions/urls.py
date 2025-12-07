from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict_car_price/', views.predict_car_price, name='predict_car_price'),
]

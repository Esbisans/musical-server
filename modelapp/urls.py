from django.urls import path

from . import views

urlpatterns = [
    path("index/", views.index, name="index"),
    path("local/", views.local_image, name="local_image"),
    path("predict/", views.predict_image, name="predict_image"),
    path("example/", views.example, name="example"),
]
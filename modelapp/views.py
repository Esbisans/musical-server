from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import tensorflow as tf
import numpy as np
from PIL import Image

model_path = 'model/musical-cad.h5'
image_path = 'media/11.jpg'
img_height = 224
img_width = 224
class_names = ['armonica', 'bajo', 'bateria', 'cajon', 'congas', 'guitarra_acustica', 'guitarra_electrica', 'piano', 'trompeta', 'violin']

def index(request):
    #return HttpResponse("Hello, world. You're at the polls index.")
    data = {
        'name': 'Hola mundo'
    }
    return JsonResponse(data)

def local_image(request):


    model = tf.keras.models.load_model(model_path)

    # Cargar la imagen en TensorFlow
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img_array = tf.expand_dims(img, 0)  # Crear un lote (batch)

    # Realizar la predicción con el modelo
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    response = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }

    #return JsonResponse({'message': 'Modelo cargado exitosamente'})
    return JsonResponse(response)


def predict_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        # Obtener la imagen enviada desde el frontend
        image = request.FILES['image']

        # Cargar modelo 
        model = tf.keras.models.load_model(model_path)

        # Cargar la imagen en TensorFlow
        img = Image.open(image)
        img = img.resize((img_width, img_height))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Realizar la predicción con el modelo
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        response = {
            'predicted_class': predicted_class,
            'confidence': confidence
        }

        return JsonResponse(response)

    return JsonResponse({'error': 'Se esperaba una imagen en la solicitud POST.'})

def example(request):
    if request.method == 'POST':
        # Obtener los datos enviados en la solicitud POST
        name = request.POST.get('name')
        description = request.POST.get('description')

        # Realizar alguna lógica con los datos recibidos
        # En este caso, simplemente verificamos que se haya enviado el nombre 'hola'
        if name == 'hola':
            response_data = {
                'name': 'conexión exitosa'
            }
        else:
            response_data = {
                'name': 'conexión fallida'
            }

        return JsonResponse(response_data)
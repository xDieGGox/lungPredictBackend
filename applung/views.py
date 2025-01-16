from django.shortcuts import render
from rest_framework import viewsets
from .serializers import PatientSerializer, PredictionSerializer
from .models import Patient, Prediction


import os
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status



# Create your views here.

class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer


class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer


# Ruta absoluta al archivo del modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directorio actual (applung)
model_path = os.path.join(BASE_DIR, 'resources', 'modelo_entrenado_efficientnetv2Normalizado.h5')

# Cargar el modelo
model = load_model(model_path)


# Etiquetas de clases (ajusta según tu modelo)
class_labels = {0: 'Class_1', 1: 'Class_2', 2: 'Class_3'}

def predict_sequence(model, image_paths, sequence_length=50, target_size=(224, 224)):
    sequence = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalizar
        sequence.append(img_array)

    # Ajustar la secuencia a la longitud esperada
    if len(sequence) < sequence_length:
        padding = [np.zeros(target_size + (3,))] * (sequence_length - len(sequence))
        sequence.extend(padding)
    elif len(sequence) > sequence_length:
        sequence = sequence[:sequence_length]

    sequence = np.expand_dims(np.array(sequence), axis=0)  # Forma (1, 50, 224, 224, 3)
    return sequence

class PredictionAPIView(APIView):
    def get(self, request):
        return Response({"message": "Ruta /applung/v1/cfc/ está activa. Use POST para enviar imágenes."}, status=status.HTTP_200_OK)
    
    def post(self, request):
        # Verifica si se enviaron archivos
        files = request.FILES.getlist('images')
        if not files or len(files) < 50:
            return Response({"error": "Debe enviar al menos 50 imágenes."}, status=status.HTTP_400_BAD_REQUEST)

        # Guardar imágenes temporalmente
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        for i, file in enumerate(files):
            temp_path = os.path.join(temp_dir, f"image_{i}.jpg")
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(file.read())
            image_paths.append(temp_path)

        try:
            # Preprocesar las imágenes y hacer la predicción
            sequence = predict_sequence(model, image_paths)
            prediction = model.predict(sequence)
            predicted_class = np.argmax(prediction, axis=1)

            result = {
                "predicted_class": class_labels[predicted_class[0]],
                "prediction_scores": prediction.tolist(),
            }
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Limpiar archivos temporales
            for path in image_paths:
                os.remove(path)
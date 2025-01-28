from django.shortcuts import render
from rest_framework import viewsets
from .serializers import PatientSerializer, PredictionSerializer, DoctorSerializer
from .models import Patient, Prediction, Doctor


import os
import tempfile
import tensorflow as tf  # <--- IMPORTANTE: Importar TensorFlow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from openai import OpenAI
from django.conf import settings
import matplotlib.pyplot as plt
import base64
from io import BytesIO



# Create your views here.

class DoctorViewSet(viewsets.ModelViewSet):
    queryset = Doctor.objects.all()
    serializer_class = DoctorSerializer

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
class_labels = {0: 'Adenocarcinoma', 1: 'Carcinoma de celulas pequeñas', 2: 'Carcinoma de células escamosas'}

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




def integrated_gradients(model, input_sequence, baseline=None, steps=50, batch_size=20):
    if baseline is None:
        baseline = np.zeros_like(input_sequence)  # Imagen baseline negra
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
    input_sequence = tf.convert_to_tensor(input_sequence, dtype=tf.float32)

    # Generar interpolaciones
    interpolated_inputs = [
        baseline + (float(i) / steps) * (input_sequence - baseline)
        for i in range(steps + 1)
    ]
    interpolated_inputs = tf.concat(interpolated_inputs, axis=0)  # Concatenar en el batch

    # Procesar por lotes
    grads_list = []
    for i in range(0, len(interpolated_inputs), batch_size):
        batch = interpolated_inputs[i : i + batch_size]
        with tf.GradientTape() as tape:
            tape.watch(batch)
            predictions = model(batch)
        grads = tape.gradient(predictions, batch)
        grads_list.append(grads)

    grads = tf.concat(grads_list, axis=0)
    grads_mean = tf.reduce_mean(grads, axis=0)  # Promediar gradientes

    integrated_grads = (input_sequence - baseline) * grads_mean
    return integrated_grads.numpy()




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

            # Crear el cliente de OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)

            prompt = (
                f"Eres un asistente médico especializado en inteligencia artificial que ayuda a interpretar resultados de diagnóstico relacionados con el cáncer de pulmón. "
                f"A continuación, se presenta una predicción realizada por un modelo de IA basada en imágenes médicas.\n\n"
                f"El modelo predice una de las siguientes tres clases: Adenocarcinoma, Carcinoma de células pequeñas y Carcinoma de células escamosas. "
                f"La predicción actual indica que la clase más probable es: {result['predicted_class']}. "
                f"Las puntuaciones obtenidas por el modelo son:  {result['prediction_scores']}.\n\n"
                "Por favor, proporciona un informe detallado que:\n"
                "1. Explique qué es la clase {predicted_class} en términos médicos.\n"
                "2. Describa las características principales de esta enfermedad (como causas, síntomas, y grupos de riesgo).\n"
                "3. Incluya recomendaciones o posibles pasos a seguir en el diagnóstico o tratamiento, dejando claro que esto es un apoyo al diagnóstico basado en inteligencia artificial y no un diagnóstico definitivo.\n"
                "4. Opcionalmente, mencione las otras dos clases para brindar contexto y comparación.\n\n"
                "Empieza el reporte con una introducción que explique que este informe es generado como apoyo al diagnóstico médico basado en inteligencia artificial.\n\n"
                "Todo esto hazlo en un máximo de 550 palabras"
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente médico experto en diagnósticos basados en inteligencia artificial. Responde de manera profesional y clara."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=600,
            )
            
            openai_response = response.choices[0].message

            # Añadir la respuesta de OpenAI al resultado
            result["openai_explanation"] = openai_response

            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Limpiar archivos temporales
            for path in image_paths:
                os.remove(path)





class HeatmapAPIView(APIView):
    def post(self, request):
        # Verifica si se enviaron archivos
        files = request.FILES.getlist('images')
        if not files or len(files) < 1:
            return Response({"error": "Debe enviar al menos una imagen."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Guardar imágenes temporalmente y procesarlas
            temp_dir = tempfile.mkdtemp()
            image_paths = []

            for i, file in enumerate(files):
                temp_path = os.path.join(temp_dir, f"image_{i}.jpg")
                with open(temp_path, 'wb') as temp_file:
                    for chunk in file.chunks():
                        temp_file.write(chunk)
                image_paths.append(temp_path)

            # Preprocesar las imágenes en una secuencia
            sequence = predict_sequence(model, image_paths)

            # Calcular gradientes integrados
            gradients = integrated_gradients(model, sequence, steps=5, batch_size=2)

            # Seleccionar frames específicos (10 y 37)
            selected_frames = [0]  # Índices 0-basados
            heatmap_images = {}
            for frame_index in selected_frames:
                # Crear heatmap para el frame
                heatmap = np.mean(gradients[0, frame_index], axis=-1)  # Promediar sobre canales RGB
                heatmap = np.maximum(heatmap, 0)  # Eliminar valores negativos
                heatmap /= np.max(heatmap)  # Normalizar entre 0 y 1

                # Superponer heatmap sobre la imagen original
                plt.figure()
                plt.imshow(sequence[0, frame_index])  # Imagen original
                plt.imshow(heatmap, cmap='hot', alpha=0.5)  # Heatmap
                plt.colorbar()
                plt.title(f"Mapa de calor - Frame {frame_index + 1}")

                # Guardar la imagen como base64
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                heatmap_images[f"frame_{frame_index + 1}"] = heatmap_base64

            # Limpiar archivos temporales
            for path in image_paths:
                os.remove(path)
            os.rmdir(temp_dir)

            # Responder con los heatmaps
            return Response({"heatmaps": heatmap_images}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



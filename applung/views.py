from django.shortcuts import render
from rest_framework import viewsets
from .serializers import PatientSerializer, PredictionSerializer, DoctorSerializer, PatientImageReportSerializer
from .models import Patient, Prediction, Doctor, PatientImageReport


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
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError


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

class PatientImageReportViewSet(viewsets.ModelViewSet):
    queryset = PatientImageReport.objects.all()
    serializer_class = PatientImageReportSerializer


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



import numpy as np

def integrated_gradients(model, input_sequence, baseline=None, steps=50, batch_size=20):
    if baseline is None:
        baseline = tf.zeros_like(input_sequence, dtype=tf.float32)  # 🔥 Asegurar float32

    # Convertir a tensores asegurando float32
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
    input_sequence = tf.convert_to_tensor(input_sequence, dtype=tf.float32)  

    # 🔥 Optimización: Generar interpolaciones con tf.linspace y asegurar forma correcta
    alpha = tf.linspace(0.0, 1.0, steps + 1)  # Generar valores interpolados
    alpha = tf.cast(alpha, dtype=tf.float32)  # Convertir a float32

    # 🔥 Expandir dimensiones correctamente para que coincidan con input_sequence
    alpha = tf.reshape(alpha, [steps + 1] + [1] * (len(input_sequence.shape) - 1))  

    # 🔥 Ahora las dimensiones coinciden en la operación de interpolación
    interpolated_inputs = baseline + alpha * (input_sequence - baseline)
    interpolated_inputs = tf.reshape(interpolated_inputs, [-1] + list(input_sequence.shape[1:]))

    # Procesar por lotes
    grads_list = []
    for i in range(0, tf.shape(interpolated_inputs)[0], batch_size):
        batch = interpolated_inputs[i : i + batch_size]
        with tf.GradientTape() as tape:
            tape.watch(batch)
            predictions = model(batch)
        grads = tape.gradient(predictions, batch)
        grads_list.append(grads)

    grads = tf.concat(grads_list, axis=0)
    grads_mean = tf.reduce_mean(grads, axis=0)  # Promediar gradientes

    # Calcular los gradientes integrados
    integrated_grads = (input_sequence - baseline) * grads_mean
    return integrated_grads.numpy()





def integrated_gradients_antiguo(model, input_sequence, baseline=None, steps=50, batch_size=20):
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
        patient_id = request.data.get('patient_id')
        doctor_id = request.data.get('doctor_id')
        files = request.FILES.getlist('images')
        

        if not patient_id or not doctor_id:
            return Response({"error": "Debe proporcionar el ID del paciente y el ID del médico."}, status=status.HTTP_400_BAD_REQUEST)


        # Verificar que el paciente y el médico existen
        patient = get_object_or_404(Patient, id=patient_id)
        doctor = get_object_or_404(Doctor, id=doctor_id)       

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
            
            openai_response = response.choices[0].message.content

            # Guardar la predicción en la base de datos antes de responder
            prediction_instance = Prediction.objects.create(
                class_predict=result['predicted_class'],
                prediagnostic_report=openai_response,
                doctor_id=doctor.id,
                patient_id=patient.id
            )

            # Añadir la respuesta de OpenAI al resultado
            result["openai_explanation"] = openai_response

            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Limpiar archivos temporales
            for path in image_paths:
                os.remove(path)





import os
import tempfile
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 🔥 Desactiva Tkinter y usa Agg para evitar errores en hilos
import matplotlib.pyplot as plt
from io import BytesIO
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView

class AntiguoHeatmapAPIView(APIView):
    def post(self, request):
        # Verifica si se enviaron archivos
        files = request.FILES.getlist('images')
        print(f"🔍 Imágenes recibidas: {len(files)}")
        if not files or len(files) < 1:
            return Response({"error": "Debe enviar al menos una imagen."}, status=status.HTTP_400_BAD_REQUEST)


        try:
            # Crear un directorio temporal para almacenar las imágenes
            temp_dir = tempfile.mkdtemp()
            heatmap_results = []

            for i, file in enumerate(files):
                temp_path = os.path.join(temp_dir, f"image_{i}.jpg")

                # Guardar la imagen temporalmente
                with open(temp_path, 'wb') as temp_file:
                    for chunk in file.chunks():
                        temp_file.write(chunk)

                # Preprocesar la imagen (asumiendo que tu modelo espera una lista de imágenes)
                sequence = predict_sequence(model, [temp_path])

                # Calcular gradientes integrados
                gradients = integrated_gradients(model, sequence, steps=1, batch_size=2)

                # Seleccionar frames específicos (por ejemplo, solo el primer frame)
                frame_index = 0  # Solo un frame para cada imagen
                heatmap = np.mean(gradients[0, frame_index], axis=-1)  # Promediar sobre canales RGB
                heatmap = np.maximum(heatmap, 0)  # Eliminar valores negativos
                heatmap /= np.max(heatmap)  # Normalizar entre 0 y 1

                # Superponer heatmap sobre la imagen original
                plt.figure()
                plt.imshow(sequence[0, frame_index])  # Imagen original
                plt.imshow(heatmap, cmap='hot', alpha=0.5)  # Heatmap
                plt.colorbar()
                plt.title(f"Mapa de calor - Imagen {i+1}")

                # Convertir imagen a base64
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                # Agregar resultado a la lista
                heatmap_results.append({
                    "image_index": i + 1,
                    "heatmap": heatmap_base64
                })

                # Eliminar imagen temporal
                os.remove(temp_path)

            # Eliminar el directorio temporal
            os.rmdir(temp_dir)

            # Responder con la lista de heatmaps
            return Response({"heatmaps": heatmap_results}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)








class HeatmapAPIView(APIView):
    def post(self, request):
        # Obtener patient_id del request
        patient_id = request.data.get('patient_id')
        if not patient_id:
            return Response({"error": "Debe proporcionar el ID del paciente."}, status=status.HTTP_400_BAD_REQUEST)

        # Verificar que el paciente existe
        patient = get_object_or_404(Patient, id=patient_id)

        # Verifica si se enviaron archivos
        files = request.FILES.getlist('images')
        if not files or len(files) < 1:
            return Response({"error": "Debe enviar al menos una imagen."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Crear un directorio temporal para almacenar las imágenes
            temp_dir = tempfile.mkdtemp()
            image_base64_list = []
            heatmap_base64_list = []

            for i, file in enumerate(files[:2]):  # Solo permitimos hasta 2 imágenes
                temp_path = os.path.join(temp_dir, f"image_{i}.jpg")

                # Guardar la imagen temporalmente
                with open(temp_path, 'wb') as temp_file:
                    for chunk in file.chunks():
                        temp_file.write(chunk)

                # Convertir imagen a base64
                with open(temp_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    image_base64_list.append(encoded_string)

                # Preprocesar la imagen y calcular gradientes integrados
                sequence = predict_sequence(model, [temp_path])
                gradients = integrated_gradients(model, sequence, steps=1, batch_size=2)

                # Seleccionar el primer frame y calcular el heatmap
                frame_index = 0
                heatmap = np.mean(gradients[0, frame_index], axis=-1)  # Promediar sobre canales RGB
                heatmap = np.maximum(heatmap, 0)  # Eliminar valores negativos
                heatmap /= np.max(heatmap)  # Normalizar entre 0 y 1

                # Superponer heatmap sobre la imagen original
                plt.figure()
                plt.imshow(sequence[0, frame_index])  # Imagen original
                plt.imshow(heatmap, cmap='hot', alpha=0.5)  # Heatmap
                plt.colorbar()
                plt.title(f"Mapa de calor - Imagen {i+1}")

                # Convertir heatmap a base64
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()

                heatmap_base64_list.append(heatmap_base64)

                # Eliminar imagen temporal
                os.remove(temp_path)

            # Eliminar el directorio temporal
            os.rmdir(temp_dir)

            # Crear el cliente de OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)

            # Generar el prompt para la API de OpenAI
            prompt = (
                "Eres un asistente médico especializado en análisis de imágenes médicas. "
                "A continuación, recibirás imágenes médicas y sus gradientes integrados en formato base64. "
                "Analiza los patrones visuales y genera un reporte corto basado en los gradientes y observaciones generales.\n\n"
                "1. Identifica posibles anomalías o patrones inusuales en la imagen.\n"
                "2. Explica brevemente si la imagen muestra alguna característica notable.\n"
                "3. Proporciona un informe de no más de 100 palabras con conclusiones breves y directas.\n\n"
                "Imagen en base64:\n" + image_base64_list[0] if image_base64_list else ""
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis de imágenes médicas."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
            )

            openai_report = response.choices[0].message.content

            # Guardar en la base de datos
            patient_report = PatientImageReport.objects.create(
                patient=patient,
                image_1=image_base64_list[0] if len(image_base64_list) > 0 else None,
                image_2=image_base64_list[1] if len(image_base64_list) > 1 else None,
                report=openai_report
            )

            return Response({
                "patient_id": patient.id,
                "report": openai_report,
                "images_stored": len(image_base64_list),
                "image_1_base64": image_base64_list[0] if len(image_base64_list) > 0 else None,
                "image_2_base64": image_base64_list[1] if len(image_base64_list) > 1 else None,
                "heatmap_1_base64": heatmap_base64_list[0] if len(heatmap_base64_list) > 0 else None,
                "heatmap_2_base64": heatmap_base64_list[1] if len(heatmap_base64_list) > 1 else None
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)










from rest_framework.generics import RetrieveAPIView

class PatientDetailView(RetrieveAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
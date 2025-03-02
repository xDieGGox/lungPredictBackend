from rest_framework import serializers
from .models import Patient, Prediction, Doctor, PatientImageReport

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'  # Incluir todos los campos del modelo

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'

class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Doctor
        fields = '__all__'

class PatientImageReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientImageReport
        fields = '__all__'
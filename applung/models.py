from django.db import models

# Create your models here.

class Patient(models.Model):
    SEX_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]

    id = models.AutoField(primary_key=True)  
    identification = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=70) 
    last_name = models.CharField(max_length=70) 
    sex = models.CharField(max_length=1, choices=SEX_CHOICES) 
    birthday = models.DateField()

    def __str__(self):
        return f"{self.name} {self.last_name} - {self.identification}"

class Doctor(models.Model):
    id = models.AutoField(primary_key=True)  # Campo ID autoincremental
    first_name = models.CharField(max_length=70)  # Nombre del médico
    last_name = models.CharField(max_length=70)  # Apellido del médico
    email = models.EmailField(unique=True)  # Correo del médico (debe ser único)
    password = models.CharField(max_length=128)  # Contraseña del médico

    def __str__(self):
        return f"Dr. {self.first_name} {self.last_name}"

class Prediction(models.Model):
    id = models.AutoField(primary_key=True)  # Campo ID autoincremental
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)  # Relación con Patient
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)  # Relación con Doctor
    class_predict = models.CharField(max_length=200, null=True, blank=True)  # Clase de predicción (puede estar vacía)
    prediagnostic_report = models.TextField(null=True, blank=True)  # Informe de prediagnóstico con IA

    def __str__(self):
        return f"Prediction for {self.patient.name} {self.patient.last_name} by {self.doctor.first_name} {self.doctor.last_name}: {self.class_predict or 'Pending'}"
    


class PatientImageReport(models.Model):
    id = models.AutoField(primary_key=True)  # Campo ID autoincremental
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)  # Relación con Patient
    image_1 = models.TextField(null=True, blank=True)  # Imagen en base64 (puede ser nula)
    image_2 = models.TextField(null=True, blank=True)  # Imagen en base64 (puede ser nula)
    report = models.TextField(null=True, blank=True)  # Reporte basado en las imágenes

    def __str__(self):
        return f"Image Report for {self.patient.name} {self.patient.last_name}"
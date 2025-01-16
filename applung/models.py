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

class Prediction(models.Model):
    id = models.AutoField(primary_key=True)  # Campo ID autoincremental
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)  # Relación con Patient
    class_predict = models.CharField(max_length=200, null=True, blank=True)  # Clase de predicción (puede estar vacía)

    def __str__(self):
        return f"Prediction for {self.patient.name} {self.patient.last_name}: {self.class_predict or 'Pending'}"
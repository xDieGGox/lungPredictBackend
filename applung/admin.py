from django.contrib import admin
from applung.models import Patient,Prediction,Doctor

# Register your models here.
@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ('id', 'first_name', 'last_name', 'email', 'password')

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('id', 'identification', 'name', 'last_name', 'sex', 'birthday')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'class_predict')

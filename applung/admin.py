from django.contrib import admin
from applung.models import Patient,Prediction

# Register your models here.
@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('id', 'identification', 'name', 'last_name', 'sex', 'birthday')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'class_predict')

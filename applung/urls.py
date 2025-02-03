from django.urls import path, include
from rest_framework import routers
from applung import views
#from .views import PatientList, PredictionList
from .views import PredictionAPIView, HeatmapAPIView

router = routers.DefaultRouter()
router.register(r'doctors', views.DoctorViewSet)
router.register(r'patients', views.PatientViewSet)
router.register(r'predictions', views.PredictionViewSet)
router.register(r'patientimagereports', views.PatientImageReportViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('cfc/', PredictionAPIView.as_view()),
    path('heats/', HeatmapAPIView.as_view()),
    path('patients/<int:pk>/', views.PatientDetailView.as_view(), name='patient-detail'),
]


##urlpatterns = [
    ##path('patients/', PatientList.as_view(), name='patient-list'),
    ##path('predictions/', PredictionList.as_view(), name='prediction-list'),
##]
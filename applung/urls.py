from django.urls import path, include
from rest_framework import routers
from applung import views
#from .views import PatientList, PredictionList
from .views import PredictionAPIView

router = routers.DefaultRouter()
router.register(r'patients', views.PatientViewSet)
router.register(r'predictions', views.PredictionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('cfc/', PredictionAPIView.as_view()),
]


##urlpatterns = [
    ##path('patients/', PatientList.as_view(), name='patient-list'),
    ##path('predictions/', PredictionList.as_view(), name='prediction-list'),
##]
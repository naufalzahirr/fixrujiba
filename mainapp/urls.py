from django.urls import path
from .views import (
    absensi_list, absensi_create, absensi_update, absensi_delete,
    jurnalharian_list, jurnalharian_create, jurnalharian_update, jurnalharian_delete,
    labeleddata_list, labeleddata_create, labeleddata_update, labeleddata_delete,
    regu_list, regu_create, regu_update, regu_delete,
    survey_list, survey_create, survey_update, survey_delete,
    dashboard
)

urlpatterns = [
    path('absensi/', absensi_list, name='absensi_list'),
    path('absensi/create/', absensi_create, name='absensi_create'),
    path('absensi/update/<int:pk>/', absensi_update, name='absensi_update'),
    path('absensi/delete/<int:pk>/', absensi_delete, name='absensi_delete'),
    path('jurnalharian/', jurnalharian_list, name='jurnalharian_list'),
    path('jurnalharian/create/', jurnalharian_create, name='jurnalharian_create'),
    path('jurnalharian/update/<int:pk>/', jurnalharian_update, name='jurnalharian_update'),
    path('jurnalharian/delete/<int:pk>/', jurnalharian_delete, name='jurnalharian_delete'),
    path('labeleddata/', labeleddata_list, name='labeleddata_list'),
    path('labeleddata/create/', labeleddata_create, name='labeleddata_create'),
    path('labeleddata/update/<int:pk>/', labeleddata_update, name='labeleddata_update'),
    path('labeleddata/delete/<int:pk>/', labeleddata_delete, name='labeleddata_delete'),
    path('regu/', regu_list, name='regu_list'),
    path('regu/create/', regu_create, name='regu_create'),
    path('regu/update/<int:pk>/', regu_update, name='regu_update'),
    path('regu/delete/<int:pk>/', regu_delete, name='regu_delete'),
    path('survey/', survey_list, name='survey_list'),
    path('survey/create/', survey_create, name='survey_create'),
    path('survey/update/<int:pk>/', survey_update, name='survey_update'),
    path('survey/delete/<int:pk>/', survey_delete, name='survey_delete'),
    path('dashboard/', dashboard, name='dashboard'),
]

from django import forms
from .models import Absensi, JurnalHarian, Survey, Regu, LabeledData

class AbsensiForm(forms.ModelForm):
    class Meta:
        model = Absensi
        fields = '__all__'

class JurnalHarianForm(forms.ModelForm):
    class Meta:
        model = JurnalHarian
        fields = '__all__'

class SurveyForm(forms.ModelForm):
    class Meta:
        model = Survey
        fields = '__all__'

class ReguForm(forms.ModelForm):
    class Meta:
        model = Regu
        fields = '__all__'

class LabeledDataForm(forms.ModelForm):
    class Meta:
        model = LabeledData
        fields = '__all__'

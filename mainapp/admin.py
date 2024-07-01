from django.contrib import admin
from .models import Absensi, JurnalHarian, Survey, Regu, LabeledData

admin.site.register(Absensi)
admin.site.register(JurnalHarian)
admin.site.register(Survey)
admin.site.register(Regu)
admin.site.register(LabeledData)

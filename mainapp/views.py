from django.shortcuts import render, redirect
from .models import Regu, Absensi, JurnalHarian, Survey, LabeledData
from django.db.models import Avg, Sum, Count, Case, When, IntegerField, Value
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import tree
import matplotlib
matplotlib.use('Agg')  # Ubah backend menjadi 'Agg'
import matplotlib.pyplot as plt
import io
import base64
import urllib
from tabulate import tabulate

def dashboard(request):
    if request.method == 'POST':
        quarter = request.POST.get('quarter')
        year = request.POST.get('year')

        # Filter and aggregate data based on the input
        regus = Regu.objects.all()
        labeled_data = []

        for regu in regus:
            absensi_records = Absensi.objects.filter(
                regu=regu, 
                tanggal__year=year, 
                tanggal__quarter=quarter
            )
            total_absensi = absensi_records.aggregate(sum_total=Sum('total'))['sum_total']
            count_absensi = absensi_records.count()
            if count_absensi > 0:
                nilai_absen = 100 - (total_absensi / count_absensi)
            else:
                nilai_absen = None

            # Calculate nilai_jurnal based on the new rules
            jurnal_harian_records = JurnalHarian.objects.filter(
                regu=regu, 
                tanggal__year=year, 
                tanggal__quarter=quarter
            ).annotate(
                point=Case(
                    When(nilai=2, then=Value(100)),
                    When(nilai=1, then=Value(50)),
                    When(nilai=0, then=Value(0)),
                    default=Value(0),
                    output_field=IntegerField(),
                )
            )
            nilai_jurnal = jurnal_harian_records.aggregate(avg_point=Avg('point'))['avg_point']
            
            nilai_kuisioner = Survey.objects.filter(
                regu=regu, 
                tanggal__year=year, 
                tanggal__quarter=quarter
            ).aggregate(avg_kuisioner=Avg('point_regu'))['avg_kuisioner']
            
            labeled_data.append({
                'REGU': regu.nama_regu,
                'NILAI_ABSEN': float(nilai_absen) if nilai_absen is not None else None,
                'NILAI_JURNAL': float(nilai_jurnal) if nilai_jurnal is not None else None,
                'NILAI_KUISIONER': float(nilai_kuisioner) if nilai_kuisioner is not None else None,
                'STATUS': None  # Will be filled after training
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(labeled_data)

        # Load training data
        labeled_df = pd.DataFrame(list(LabeledData.objects.all().values('regu', 'nilai_absen', 'nilai_jurnal', 'nilai_kuisioner', 'status')))
        labeled_df.columns = ['REGU', 'NILAI_ABSEN', 'NILAI_JURNAL', 'NILAI_KUISIONER', 'STATUS']

        # Training RandomForest model
        penilaian_columns = ['NILAI_ABSEN', 'NILAI_JURNAL', 'NILAI_KUISIONER']
        label_encoder = LabelEncoder()
        labeled_df['STATUS'] = label_encoder.fit_transform(labeled_df['STATUS'])
        X_train = labeled_df[penilaian_columns]
        y_train = labeled_df['STATUS']
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on the test data
        X_test = df[penilaian_columns]
        X_test = imputer.transform(X_test)
        y_pred = model.predict(X_test)
        df['STATUS'] = label_encoder.inverse_transform(y_pred)

        # Calculate accuracy on the training set
        y_train_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_train_pred)
        
        # Classification report
        report = classification_report(y_train, y_train_pred, target_names=label_encoder.classes_)

        # Plot tree
        fig, ax = plt.subplots(figsize=(15, 10))
        tree.plot_tree(model.estimators_[0], feature_names=penilaian_columns, class_names=label_encoder.classes_, filled=True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image_png = buf.getvalue()
        buf.close()
        image_base64 = base64.b64encode(image_png).decode('utf-8')

        # Assign rank based on STATUS
        status_order = {'Terbaik': 1, 'Baik': 2, 'Cukup': 3, 'Buruk': 4, 'Terburuk': 5}
        df['RANK'] = df['STATUS'].map(status_order).rank(method='dense').astype(int)
        df = df.sort_values('RANK')

        # Prepare the top results in a markdown-like format
        df_hasil = df.head(3)
        top_regu_terbaik = tabulate(df_hasil[['RANK', 'REGU', 'NILAI_ABSEN', 'NILAI_JURNAL', 'NILAI_KUISIONER']], headers='keys', tablefmt='pipe', floatfmt=".2f")

        # Render the result on the page
        context = {
            'data': df.to_dict(orient='records'),
            'accuracy': accuracy,
            'report': report,
            'tree_image': image_base64,
            'top_regu_terbaik': top_regu_terbaik
        }
        return render(request, 'mainapp/dashboard.html', context)
    
    return render(request, 'mainapp/dashboard.html')



# Regu Views
def regu_list(request):
    regus = Regu.objects.all()
    return render(request, 'mainapp/regu_list.html', {'regus': regus})

def regu_create(request):
    if request.method == 'POST':
        form = ReguForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('regu_list')
    else:
        form = ReguForm()
    return render(request, 'mainapp/regu_form.html', {'form': form})

def regu_update(request, pk):
    regu = Regu.objects.get(pk=pk)
    if request.method == 'POST':
        form = ReguForm(request.POST, instance=regu)
        if form.is_valid():
            form.save()
            return redirect('regu_list')
    else:
        form = ReguForm(instance=regu)
    return render(request, 'mainapp/regu_form.html', {'form': form})

def regu_delete(request, pk):
    regu = Regu.objects.get(pk=pk)
    if request.method == 'POST':
        regu.delete()
        return redirect('regu_list')
    return render(request, 'mainapp/regu_confirm_delete.html', {'object': regu})

# Absensi Views
def absensi_list(request):
    absensis = Absensi.objects.all()
    return render(request, 'mainapp/absensi_list.html', {'absensis': absensis})

def absensi_create(request):
    if request.method == 'POST':
        form = AbsensiForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('absensi_list')
    else:
        form = AbsensiForm()
    return render(request, 'mainapp/absensi_form.html', {'form': form})

def absensi_update(request, pk):
    absensi = Absensi.objects.get(pk=pk)
    if request.method == 'POST':
        form = AbsensiForm(request.POST, instance=absensi)
        if form.is_valid():
            form.save()
            return redirect('absensi_list')
    else:
        form = AbsensiForm(instance=absensi)
    return render(request, 'mainapp/absensi_form.html', {'form': form})

def absensi_delete(request, pk):
    absensi = Absensi.objects.get(pk=pk)
    if request.method == 'POST':
        absensi.delete()
        return redirect('absensi_list')
    return render(request, 'mainapp/absensi_confirm_delete.html', {'object': absensi})

# JurnalHarian Views
def jurnalharian_list(request):
    jurnalharians = JurnalHarian.objects.all()
    return render(request, 'mainapp/jurnalharian_list.html', {'jurnalharians': jurnalharians})

def jurnalharian_create(request):
    if request.method == 'POST':
        form = JurnalHarianForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('jurnalharian_list')
    else:
        form = JurnalHarianForm()
    return render(request, 'mainapp/jurnalharian_form.html', {'form': form})

def jurnalharian_update(request, pk):
    jurnalharian = JurnalHarian.objects.get(pk=pk)
    if request.method == 'POST':
        form = JurnalHarianForm(request.POST, instance=jurnalharian)
        if form.is_valid():
            form.save()
            return redirect('jurnalharian_list')
    else:
        form = JurnalHarianForm(instance=jurnalharian)
    return render(request, 'mainapp/jurnalharian_form.html', {'form': form})

def jurnalharian_delete(request, pk):
    jurnalharian = JurnalHarian.objects.get(pk=pk)
    if request.method == 'POST':
        jurnalharian.delete()
        return redirect('jurnalharian_list')
    return render(request, 'mainapp/jurnalharian_confirm_delete.html', {'object': jurnalharian})

# Survey Views
def survey_list(request):
    surveys = Survey.objects.all()
    return render(request, 'mainapp/survey_list.html', {'surveys': surveys})

def survey_create(request):
    if request.method == 'POST':
        form = SurveyForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('survey_list')
    else:
        form = SurveyForm()
    return render(request, 'mainapp/survey_form.html', {'form': form})

def survey_update(request, pk):
    survey = Survey.objects.get(pk=pk)
    if request.method == 'POST':
        form = SurveyForm(request.POST, instance=survey)
        if form.is_valid():
            form.save()
            return redirect('survey_list')
    else:
        form = SurveyForm(instance=survey)
    return render(request, 'mainapp/survey_form.html', {'form': form})

def survey_delete(request, pk):
    survey = Survey.objects.get(pk=pk)
    if request.method == 'POST':
        survey.delete()
        return redirect('survey_list')
    return render(request, 'mainapp/survey_confirm_delete.html', {'object': survey})

# LabeledData Views
def labeleddata_list(request):
    labeleddatas = LabeledData.objects.all()
    return render(request, 'mainapp/labeleddata_list.html', {'labeleddatas': labeleddatas})

def labeleddata_create(request):
    if request.method == 'POST':
        form = LabeledDataForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('labeleddata_list')
    else:
        form = LabeledDataForm()
    return render(request, 'mainapp/labeleddata_form.html', {'form': form})

def labeleddata_update(request, pk):
    labeleddata = LabeledData.objects.get(pk=pk)
    if request.method == 'POST':
        form = LabeledDataForm(request.POST, instance=labeleddata)
        if form.is_valid():
            form.save()
            return redirect('labeleddata_list')
    else:
        form = LabeledDataForm(instance=labeleddata)
    return render(request, 'mainapp/labeleddata_form.html', {'form': form})

def labeleddata_delete(request, pk):
    labeleddata = LabeledData.objects.get(pk=pk)
    if request.method == 'POST':
        labeleddata.delete()
        return redirect('labeleddata_list')
    return render(request, 'mainapp/labeleddata_confirm_delete.html', {'object': labeleddata})

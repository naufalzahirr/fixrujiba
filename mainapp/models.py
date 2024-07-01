from django.db import models

class Regu(models.Model):
    nama_regu = models.CharField(max_length=100)

    def __str__(self):
        return self.nama_regu

class Absensi(models.Model):
    nip = models.CharField(max_length=20)
    tanggal = models.DateField()
    nama = models.CharField(max_length=100)
    unit_kerja = models.CharField(max_length=100)
    regu = models.ForeignKey(Regu, on_delete=models.CASCADE)
    terlambat = models.IntegerField(default=0)
    pulang_cepat = models.IntegerField(default=0)
    terlambat_menit = models.IntegerField(default=0)
    pulang_cepat_menit = models.IntegerField(default=0)
    tanpa_keterangan = models.IntegerField(default=0)
    terlambat_izin = models.IntegerField(default=0)
    pulang_cepat_izin = models.IntegerField(default=0)
    lupa_absen_masuk = models.IntegerField(default=0)
    lupa_absen_pulang = models.IntegerField(default=0)
    full = models.IntegerField(default=0)
    half = models.IntegerField(default=0)
    total = models.IntegerField(default=0)

    def __str__(self):
        return f'{self.nama} - {self.nip}'

class JurnalHarian(models.Model):
    nama = models.CharField(max_length=100)
    tanggal = models.DateField()
    skp_tahunan = models.CharField(max_length=100)
    regu = models.ForeignKey(Regu, on_delete=models.CASCADE)
    jurnal_harian = models.TextField()
    jumlah = models.IntegerField()
    satuan = models.CharField(max_length=50)
    jam_mulai = models.TimeField()
    jam_selesai = models.TimeField()
    nilai = models.DecimalField(max_digits=5, decimal_places=2)
    komentar = models.TextField()
    tanggal_isi = models.DateField(auto_now_add=True)
    lampiran = models.FileField(upload_to='lampiran/')

    def __str__(self):
        return f'{self.nama} - {self.tanggal}'

class Survey(models.Model):
    regu = models.ForeignKey(Regu, on_delete=models.CASCADE)
    tanggal = models.DateField()
    point_regu = models.IntegerField()

    def __str__(self):
        return f'{self.regu} - {self.point_regu}'

class LabeledData(models.Model):
    regu = models.CharField(max_length=100)
    nilai_absen = models.DecimalField(max_digits=5, decimal_places=2)
    nilai_jurnal = models.DecimalField(max_digits=5, decimal_places=2)
    nilai_kuisioner = models.DecimalField(max_digits=5, decimal_places=2)
    status = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.regu} - {self.status}'

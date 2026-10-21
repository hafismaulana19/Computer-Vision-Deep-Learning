# 🤖 P2 Transfer Learning — RET503

### Computer Vision and Deep Learning
**Mata Kuliah:** RET503 — Computer Vision and Deep Learning  
**Program Studi:** Teknologi Rekayasa Robotika  
**Institusi:** Politeknik Negeri Batam  

> Implementasi transfer learning dan fine-tuning untuk klasifikasi komponen robot menggunakan dataset **MG90S** dan **PCA9685**.

---

## 📌 Deskripsi

Project ini merupakan implementasi **Transfer Learning dan Fine-Tuning Model Visi** untuk tugas P2 mata kuliah **RET503 Computer Vision and Deep Learning**.

Model digunakan untuk melakukan klasifikasi citra dua jenis komponen robot:

- 🔧 **MG90S Servo**
- 🔌 **PCA9685 Servo Driver**

Eksperimen utama menggunakan **ResNet18 pretrained ImageNet** dengan tiga strategi pembelajaran:

1. **Feature Extraction**
2. **Partial Fine-Tuning**
3. **Training from Scratch**

Selain eksperimen utama tersebut, project juga dilengkapi dengan implementasi **MobileNetV2** dan pengujian kamera secara real-time.

Tujuan akhirnya adalah mendapatkan model yang memiliki kombinasi **akurasi yang baik dan latency yang sesuai untuk aplikasi robotika**.

---

💻 Instalasi dan Cara Menjalankan

1. Persiapan Environment

Pastikan Python sudah terinstal. Buka terminal pada folder Transfer-Learning, kemudian jalankan:

python -m pip install -r requirements.txt

2. Training MobileNetV2

Jalankan script training sesuai dataset dan konfigurasi yang digunakan:

python src/train.py

Untuk eksperimen MobileNetV2 V2:

python src/train_v2.py

3. Training ResNet18

Untuk membandingkan Feature Extraction, Partial Fine-Tuning, dan Training from Scratch, jalankan:

python src/train_resnet18.py

4. Evaluasi Model

Jalankan script evaluasi sesuai model dan dataset yang ingin diuji. Contoh:

python src/evaluate.py
python src/evaluate_mobilenetv2_external.py
python src/evaluate_resnet18_external.py

5. Pengujian Kamera

Untuk pengujian klasifikasi melalui kamera, jalankan:

python src/detect.py

atau:

python src/test.py

Catatan: Pastikan dataset, model hasil training, dan file nama kelas tersedia sesuai konfigurasi masing-masing script sebelum menjalankan perintah di atas.

---

## 🎯 Tujuan

Project ini bertujuan untuk:

- Memahami konsep transfer learning pada computer vision.
- Menggunakan model CNN pretrained untuk klasifikasi objek.
- Membandingkan Feature Extraction, Partial Fine-Tuning, dan Training from Scratch.
- Menganalisis pengaruh pretrained weights terhadap performa model.
- Membandingkan performa ResNet18 dan MobileNetV2.
- Mengukur performa model berdasarkan accuracy dan inference latency.
- Menguji model menggunakan data external yang tidak digunakan selama training.
- Mengimplementasikan model untuk real-time camera classification.

---

## 🧠 Konsep Transfer Learning

Transfer learning memanfaatkan pengetahuan yang telah diperoleh model dari
dataset sumber untuk menyelesaikan tugas baru pada dataset target.

Pada project ini, model **ResNet18 pretrained ImageNet** digunakan sebagai
backbone untuk klasifikasi dua jenis komponen robot, yaitu MG90S dan PCA9685.

```text
                 PRETRAINED MODEL
                    ImageNet
                       │
                       ▼
                ┌─────────────┐
                │  ResNet18   │
                └─────────────┘
                       │
                       ▼
             Dataset Komponen Robot
                       │
              ┌────────┴────────┐
              ▼                 ▼
           MG90S             PCA9685
```
---

## 📊 Analisis Hasil ResNet18

Berdasarkan hasil eksperimen tiga strategi pembelajaran pada ResNet18,
diperoleh hasil sebagai berikut:

| Metode | Accuracy |
|---|---:|
| Feature Extraction | 97.50% |
| Partial Fine-Tuning | 97.50% |
| Training from Scratch | 75.00% |

Feature Extraction dan Partial Fine-Tuning menghasilkan accuracy yang sama,
yaitu **97.50%**, sedangkan Training from Scratch menghasilkan accuracy
sebesar **75.00%**.

Hasil tersebut menunjukkan bahwa penggunaan pretrained weights dari ImageNet
memberikan keuntungan pada dataset yang digunakan. Dengan memanfaatkan fitur
yang telah dipelajari sebelumnya, model dapat memperoleh performa yang lebih
baik dibandingkan model yang dilatih dari awal.

Pada eksperimen ini, Partial Fine-Tuning belum memberikan peningkatan accuracy
dibandingkan Feature Extraction. Hal tersebut menunjukkan bahwa membuka
sebagian layer backbone belum memberikan keuntungan tambahan pada dataset dan
konfigurasi training yang digunakan.

Sebaliknya, Training from Scratch menghasilkan accuracy yang lebih rendah.
Hal ini menunjukkan bahwa pada dataset dengan jumlah data yang relatif
terbatas, penggunaan pretrained model dapat membantu proses pembelajaran
fitur visual secara lebih efektif.

Dengan demikian, hasil eksperimen ResNet18 menunjukkan bahwa pendekatan
berbasis transfer learning lebih efektif dibandingkan Training from Scratch
untuk dataset yang digunakan pada project ini.

---

## 🧪 Analisis External Testing MobileNetV2

Setelah proses training, model MobileNetV2 V2 diuji menggunakan dataset
external yang tidak digunakan selama proses training.

Dataset external terdiri dari:

| Kelas | Jumlah Data |
|---|---:|
| MG90S | 20 |
| PCA9685 | 20 |
| **Total** | **40** |

Hasil external testing:

| Parameter | Hasil |
|---|---:|
| Total images | 40 |
| Correct prediction | 40 |
| Wrong prediction | 0 |
| External Accuracy | **100.00%** |
| MG90S | **100.00% (20/20)** |
| PCA9685 | **100.00% (20/20)** |
| Average Latency | **84.136 ms/image** |

Model berhasil mengklasifikasikan seluruh 40 gambar external test dengan benar.
Tidak terdapat kesalahan klasifikasi pada kedua kelas.

Hasil tersebut menunjukkan bahwa model MobileNetV2 V2 mampu melakukan
generalisasi dengan baik terhadap data yang tidak digunakan selama training.

Namun, nilai accuracy yang tinggi pada external test tetap perlu
diinterpretasikan berdasarkan ukuran dataset. External test yang digunakan
berjumlah 40 gambar, sehingga hasil 100% menunjukkan bahwa seluruh sampel
pada dataset external berhasil diklasifikasikan dengan benar, tetapi belum
dapat secara mutlak merepresentasikan performa model pada seluruh kondisi
lingkungan nyata.

Pengujian ini kemudian dilanjutkan dengan camera test untuk melihat kemampuan
model dalam melakukan klasifikasi secara real-time.

---

### Metodologi Pengukuran Latensi

Pengukuran latensi dilakukan menggunakan fungsi `time.perf_counter()` untuk mencatat waktu sebelum dan sesudah pemanggilan `model.predict()` pada setiap gambar. Selisih waktu tersebut dikonversi ke milidetik (ms) dan dicatat sebagai latensi inferensi per gambar.

Pengujian dilakukan terhadap 40 gambar dari dataset external, yang terdiri dari 20 gambar MG90S dan 20 gambar PCA9685. Rata-rata latensi dihitung dari seluruh hasil pengukuran pada gambar yang diuji.

Perlu diperhatikan bahwa pengukuran mencakup waktu pemanggilan `model.predict()`, sehingga hasilnya dapat dipengaruhi oleh overhead pemanggilan fungsi dan proses inisialisasi pada inferensi pertama. Oleh karena itu, nilai latensi ini merepresentasikan hasil pengukuran pada konfigurasi eksperimen yang digunakan, bukan murni waktu komputasi model pada kondisi yang telah dioptimalkan.

Hasil pengukuran digunakan sebagai referensi untuk membandingkan performa model dalam eksperimen ini. Perbandingan latensi sebaiknya mempertimbangkan konsistensi perangkat keras, framework, dan metode pengukuran yang digunakan.

---

## 📷 Analisis Real-Time Camera

Model MobileNetV2 V2 juga diuji menggunakan kamera secara real-time untuk
melihat kemampuan model dalam melakukan klasifikasi langsung terhadap objek.

Beberapa hasil pengujian kamera menunjukkan:

| Objek | Confidence |
|---|---:|
| PCA9685 | 99.77% |
| MG90S | 97.37% |
| MG90S | 99.38% |
| PCA9685 | 96.91% |

Model dapat mengenali kedua objek dengan confidence yang tinggi pada kondisi
pengujian kamera.

Hasil tersebut menunjukkan bahwa model tidak hanya dapat bekerja pada dataset
external, tetapi juga dapat digunakan untuk melakukan klasifikasi objek secara
langsung melalui kamera.

Meskipun demikian, performa real-time dapat dipengaruhi oleh beberapa faktor,
seperti pencahayaan, posisi objek, jarak kamera, sudut pengambilan gambar,
background, dan kondisi objek.

---

## ⚖️ Perbandingan Model

Hasil akhir perbandingan model menunjukkan:

| Model | Accuracy | Latency |
|---|---:|---:|
| MobileNetV2 | **100.00%** | 85.225 ms |
| ResNet18 Feature Extraction | 97.50% | 5.658 ms |
| ResNet18 Partial Fine-Tuning | 97.50% | 0.673 ms |
| ResNet18 Scratch | 75.00% | 0.638 ms |

MobileNetV2 menghasilkan accuracy tertinggi, yaitu **100.00% pada external
test** dengan 40 gambar. Namun, hasil tersebut perlu dipahami berdasarkan
ukuran dataset external yang relatif kecil dan tidak dapat secara langsung
digeneralisasikan ke seluruh kondisi lingkungan nyata.

Pada eksperimen ResNet18, Feature Extraction dan Partial Fine-Tuning
menghasilkan accuracy yang sama, yaitu **97.50%**. Sementara itu, Training
from Scratch hanya mencapai **75.00%**. Hasil ini memperlihatkan bahwa
penggunaan pretrained weights memberikan keuntungan pada dataset yang
digunakan dibandingkan melatih model dari awal.

Dari sisi latency, **ResNet18 Scratch memiliki latency terendah sebesar
0.638 ms**, diikuti oleh Partial Fine-Tuning sebesar 0.673 ms dan Feature
Extraction sebesar 5.658 ms. MobileNetV2 memiliki latency sebesar
85.225 ms pada pengujian yang dibandingkan.

Dengan demikian, accuracy dan latency perlu dipertimbangkan secara bersamaan.
**MobileNetV2 memberikan accuracy tertinggi sebesar 100.00%** pada external
test, sedangkan **ResNet18 Training from Scratch memberikan latency terendah
sebesar 0.638 ms**. Namun, Training from Scratch memiliki accuracy yang jauh
lebih rendah, yaitu 75.00%.

Sementara itu, **ResNet18 Partial Fine-Tuning memberikan keseimbangan yang
lebih baik antara accuracy dan latency**, dengan accuracy sebesar 97.50% dan
latency sebesar 0.673 ms. Oleh karena itu, apabila prioritas utama adalah
akurasi, MobileNetV2 menjadi pilihan terbaik pada pengujian ini. Jika
diperlukan kombinasi accuracy yang tinggi dan latency yang rendah, ResNet18
Partial Fine-Tuning dapat menjadi pilihan yang lebih sesuai untuk aplikasi
robotika.

---


## 💻 Environment Eksperimen

Eksperimen dilakukan menggunakan environment berikut:

| Komponen | Spesifikasi |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| OS | Windows |
| Python | Python 3.x |
| Framework | PyTorch |
| Computer Vision | OpenCV |
| Model | ResNet18, MobileNetV2 |
| Pretrained Weights | ImageNet |
| Hardware Acceleration | NVIDIA GPU |
| Dataset | MG90S dan PCA9685 |

GPU digunakan untuk mempercepat proses training model. Pengujian inference
dan latency dilakukan sesuai konfigurasi eksperimen yang digunakan pada
project ini.

---

## ✅ Kesimpulan

Berdasarkan seluruh eksperimen, penggunaan **transfer learning dengan
pretrained weights** memberikan keuntungan dibandingkan **Training from
Scratch** pada dataset klasifikasi MG90S dan PCA9685.

Pada ResNet18, Feature Extraction dan Partial Fine-Tuning sama-sama mencapai
**97.50% accuracy**, sedangkan Training from Scratch hanya mencapai
**75.00%**. Hal ini menunjukkan bahwa fitur yang telah dipelajari dari
ImageNet dapat membantu model memperoleh performa yang lebih baik pada dataset
target.

Pengujian external MobileNetV2 menghasilkan **100.00% accuracy** dari 40
gambar, dengan seluruh gambar berhasil diklasifikasikan dengan benar.
Pengujian kamera secara real-time juga menunjukkan bahwa model mampu
mengenali MG90S dan PCA9685 dengan confidence yang tinggi pada kondisi
pengujian.

Namun, hasil external test dan camera test tetap perlu dipandang sebagai
hasil pada kondisi pengujian yang digunakan. Performa model di lingkungan
nyata dapat berubah akibat pencahayaan, posisi objek, jarak kamera, sudut
pengambilan gambar, background, dan variasi objek.

Secara keseluruhan, project ini menunjukkan bahwa **transfer learning dapat
menjadi pendekatan yang efektif untuk klasifikasi komponen robot dengan
dataset target yang relatif terbatas**, sementara pemilihan model untuk
deployment perlu mempertimbangkan trade-off antara **accuracy, latency, dan
kebutuhan real-time sistem robotika**.
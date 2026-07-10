# CNN ile Çiçek Sınıflandırma Projesi (Flower Classification)

Bu proje, TensorFlow ve Keras kütüphaneleri kullanılarak 5 farklı çiçek türünü (Gül, Papatya, Lale, Ayçiçeği, Karahindiba) yüksek doğrulukla sınıflandırmak amacıyla geliştirilmiş bir Derin Öğrenme (Deep Learning) projesidir.

## Proje Özellikleri

- **Veri Pipeline'ı:** Veri seti `tf_records` biçiminde verimli bir şekilde yüklenmiş, bellek optimizasyonu için önbelleğe alınmış (`cache`) ve karıştırılmıştır (`shuffle`).
- **Veri Artırma (Data Augmentation):** Modelin genelleme yeteneğini artırmak amacıyla görüntüler üzerinde rastgele döndürme, yakınlaştırma ve yatay çevirme işlemleri uygulanmıştır.
- **Akıllı Eğitim Mekanizmaları (Callbacks):** Aşırı öğrenmeyi (overfitting) engellemek ve donanım kaynaklarını korumak için dinamik eğitim araçları entegre edilmiştir.
- **Görselleştirme:** Eğitim sonrasında doğruluk ve kayıp grafiklerinin analiz edilmesi amacıyla `matplotlib` entegrasyonu yapılmıştır.

---

## Model Mimarisi

Projede ardışık (Sequential) katmanlardan oluşan 3 katmanlı güçlü bir Evrişimli Sinir Ağı (CNN) mimarisi tercih edilmiştir:

| Katman Türü              | Özellikler / Filtre Sayısı    | Aktivasyon Fonksiyonu |
| :----------------------- | :---------------------------- | :-------------------- |
| **Giriş (Input)**        | 180 x 180 x 3                 | -                     |
| **Conv2D + MaxPool2D**   | 32 Filtre (3x3)               | ReLU                  |
| **Conv2D + MaxPool2D**   | 64 Filtre (3x3)               | ReLU                  |
| **Conv2D + MaxPool2D**   | 128 Filtre (3x3)              | ReLU                  |
| **Flatten**              | Vektörleştirme (51,200 boyut) | -                     |
| **Dense (Gizli Katman)** | 128 Nöron                     | ReLU                  |
| **Dropout**              | %50 Oranında Söndürme         | -                     |
| **Dense (Çıkış)**        | 5 Sınıf (Çiçek Türleri)       | Softmax / Logits      |

- **Toplam Eğitilebilir Parametre Sayısı:** 6,647,621 (~25.36 MB)
- **Optimizasyon Algoritması:** Adam Optimizer (Başlangıç Learning Rate: 0.001)
- **Kayıp Fonksiyonu:** `sparse_categorical_crossentropy`

---

## Eğitim Stratejisi (Callbacks)

Eğitimin her epoch (tur) bitiminde arka planda çalışan ve süreci optimize eden 3 temel callback mekanizması kullanılmıştır:

1. **EarlyStopping (Erken Durdurma):** Doğrulama kaybı (`val_loss`) ardışık 3 epoch boyunca daha iyi bir rekor kıramazsa eğitimi otomatik olarak sonlandırır ve modeli en başarılı olduğu ana geri sarar (`restore_best_weights=True`).
2. **ReduceLROnPlateau (Akıllı Vites Küçültme):** Doğrulama kaybı 2 epoch boyunca iyileşmezse, öğrenme oranını (Learning Rate) 0.2 çarpanı ile çarparak küçültür. Böylece model daha küçük adımlarla ince ayar yapmaya başlar.
3. **ModelCheckpoint (En İyi Anı Kaydetme):** `val_loss` değerinde ne zaman yeni bir dünya rekoru kırılsa, modelin o anki ağırlıklarını dinamik olarak `best_model.keras` dosyasına fiziksel olarak kaydeder.

---

## Eğitim Sonuçları ve Değerlendirme

Model, 10 tam epoch boyunca eğitilmiş ve elde edilen nihai performans değerleri aşağıda listelenmiştir:

- **Eğitim Doğruluğu (Training Accuracy):** %74.59
- **Doğrulama Doğruluğu (Validation Accuracy):** %74.25
- **Eğitim Kaybı (Training Loss):** 0.6769
- **Doğrulama Kaybı (Validation Loss):** 0.6736

> **Değerlendirme:** Eğitim doğruluğu (%74.59) ile doğrulama doğruluğunun (%74.25) birbirine neredeyse tamamen eşit ilerlemesi, modelimizin kesinlikle ezberleme (overfitting) yapmadığını ve yeni gördüğü çiçek fotoğraflarını başarıyla ayırt edebilecek mantığı kavradığını kanıtlamaktadır.

---

## Kurulum ve Çalıştırma

Projenin yerelde çalıştırılması için gerekli adımlar:

1. Depoyu bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/tuanaunal/flowers-cnn-classification.git](https://github.com/tuanaunal/flowers-cnn-classification.git)
   ```
2. Gerekli kütüphaneleri yükleyin:
   pip install -r requirements.txt
3. Modeli eğitmek ve grafik sonuçlarını görmek için kodu çalıştırın:
   python cnn.py

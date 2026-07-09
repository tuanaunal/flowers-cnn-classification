"""
flowers dataset:
    rgb: 224*224

CNN ile siniflandirma modeli olusturma ve problemi cozme
"""

# import libraries
from tensorflow_datasets import load # veri seti yukleme
from tensorflow.data import AUTOTUNE # veri seti optimizasyonu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, # 2D convolutional layer
    MaxPooling2D, # max pooling layer
    Flatten, # cok boyutlu veriyi tek boyutlu hale getirme
    Dense, # tam baglantili katman
    Dropout # rastgele noronlari kapatma ve overfitting'i engelleme
)

from tensorflow.keras.optimizers import Adam # optimizer
from tensorflow.keras.callbacks import (
    EarlyStopping, # erken durdurma
    ReduceLROnPlateau, # ogrenme oranini azaltma
    ModelCheckpoint # model kaydetme
)

import tensorflow as tf
import matplotlib.pyplot as plt # gorsellestirme

# veri seti yukleme
(ds_train, ds_val), ds_info = load(
    "tf_flowers", # veri seti ismi
    split = ["train[:80%]", # veri setinin %80 i egitim icin
             "train[80%:]"], # veri setinin %20 si test icin
    as_supervised=True, # veri setinin gorsel, etiket ciftinin olmasi
    with_info=True # veri seti hakkinda bilgi alma
)

print(ds_info.features) # veri seti hakkinda bilgi yazdirma
print("Number of classes:", ds_info.features['label'].num_classes)

# ornek veri gorsellestirme
# egitim setinden rastgele 3 resim ve etiket alma
fig = plt.figure(figsize = (10,5))
for i, (image, label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1, 3, i+1) # 1 satir, 3 sutun, i+1. resim
    ax.imshow(image.numpy().astype("uint8")) # resmi gorsellestirme
    ax.set_title(f"Etiket: {label.numpy()}") # etiket baslik olarak yazdirma
    ax.axis("off") # eksenleri kapatma

plt.tight_layout()
plt.show() # grafigi gosterme

# data augmentation + preprocessing

# CNN modelini olusturma

# callbacks

# derleme

# traning

# model evaluation
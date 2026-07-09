"""
flowers dataset:
    rgb: 224*224

CNN ile siniflandirma modeli olusturma ve problemi cozme

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

# ornek veri gorsellestirme

# data augmentation + preprocessing

# CNN modelini olusturma

# callbacks

# derleme

# traning

# model evaluation

"""
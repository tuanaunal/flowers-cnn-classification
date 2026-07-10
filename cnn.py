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

IMG_SIZE = (180, 180)

# data augmentation + preprocessing
def preprocess_train(image, label):
    """
    resize, random flip, brightness, contrast, crop
    normalize
    """
    image = tf.image.resize(image, IMG_SIZE) # boyutlandırma
    image = tf.image.random_flip_left_right(image) # yatay olarak rastgele cevirme
    image = tf.image.random_brightness(image, max_delta=0.1) # rastgele parlaklik
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2) # rastgele kontrast
    image = tf.image.random_crop(image, size=(160, 160, 3)) # rastgele crop
    image = tf.image.resize(image, IMG_SIZE) # tekrar boyutlandirma
    image = tf.cast(image, tf.float32)/255.0 # normalize etme
    return image, label

def preprocess_val(image, label):
    """
    resize, normalize
    """
    image = tf.image.resize(image, IMG_SIZE) # boyutlandırma
    image = tf.cast(image, tf.float32)/255.0 # normalize etme
    return image, label

# veri setini hazirlamak
ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE) # on isleme ve augmentasyon
    .shuffle(1000) # karistirma
    .batch(32) # batch boyutu
    .prefetch(AUTOTUNE) # veri setini onceden hazirlamak
)

ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE) # on isleme
    .batch(32) # batch boyutu
    .prefetch(AUTOTUNE) # veri setini onceden hazirlamak
)

# CNN modelini olusturma
model = Sequential([
    
    # Feature Extraction Layers
    Conv2D(32, (3,3), activation = "relu", input_shape = (*IMG_SIZE, 3)), # 32 filtre, 3x3 kernel, relu aktivasyon, 3 Kanal (RGB)
    MaxPooling2D((2,2)), # 2x2 max pooling
    
    Conv2D(64, (3,3), activation = "relu"), # 64 filtre, 3x3 kernel, relu aktivasyon
    MaxPooling2D((2,2)), # 2x2 max pooling
    
    Conv2D(128, (3,3), activation = "relu"), # 128 filtre, 3x3 kernel, relu aktivasyon
    MaxPooling2D((2,2)), # 2x2 max pooling
    
    # Classification Layers
    Flatten(), # cok boyutlu veriyi vektore cevir
    Dense(128, activation = "relu"),
    Dropout(0.5), # overfitting'i engellemek icin dropout
    Dense(ds_info.features["label"].num_classes, activation = "softmax") # cikis katmani, softmax aktivasyon
    
])

# callbacks
callbacks = [
    # eger val loss 3 epoch boyunca iyilesmezse egitimi durdur ve en iyi agirliklari yukle
    EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True), # erken durdurma
    
    # val loss 2 epoch boyunca iyilesmezse learning rate 0.2 carpani ile azalt
    ReduceLROnPlateau(monitor = "val_loss", factor = 0.2, patience = 2, verbose = 1, min_lr = 1e-9), # ogrenme oranini azaltma
    
    # her epoch sonunda eger model daha iyi ise kaydolur
    ModelCheckpoint("best_model.keras", save_best_only=True) # model kaydetme, en iyi modeli kaydet
]

# derleme

# traning

# model evaluation
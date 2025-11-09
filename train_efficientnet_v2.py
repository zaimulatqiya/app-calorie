import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
import os, json

# Path dataset
dataset_path = "data/dataset_makanan_indonesia"

# Data generator + augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # 🔥 ganti dari 128 ke 224
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # 🔥 ganti dari 128 ke 224
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Base model EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 🔥 Training tahap 1 (feature extraction)
history = model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[early_stop])

# 🔥 Fine-tuning tahap 2
base_model.trainable = True
for layer in base_model.layers[:-50]:  # buka 50 layer terakhir
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # 🔥 learning rate kecil
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_ft = model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[early_stop])

# Simpan model
os.makedirs("models", exist_ok=True)
model.save("models/food_model_efficientnet_v2.h5")

# Simpan mapping kelas
with open("models/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("✅ EfficientNetB0 (224x224, epoch 30) model saved as food_model_efficientnet_v2.h5")

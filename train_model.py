import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
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
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print(f"\n{'='*60}")
print(f"✅ Jumlah kelas terdeteksi: {len(train_gen.class_indices)}")
print(f"✅ Daftar kelas: {list(train_gen.class_indices.keys())}")
print(f"✅ Total training samples: {train_gen.samples}")
print(f"✅ Total validation samples: {val_gen.samples}")
print(f"{'='*60}\n")

# Validasi jumlah kelas
assert len(train_gen.class_indices) == 18, f"❌ ERROR: Expected 18 classes, got {len(train_gen.class_indices)}"

# Pretrained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # freeze base model

# Bangun model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stop]
)

# Simpan model
os.makedirs("models", exist_ok=True)
model.save("models/food_model.h5")

print("Model saved to models/food_model.h5")
print("Classes:", train_gen.class_indices)

# Simpan mapping kelas ke JSON
with open("models/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("Class indices saved to models/class_indices.json")

print("\n" + "="*60)
print("Menyimpan training history...")
with open("models/training_history_simplecnn.json", "w") as f:
    json.dump(history.history, f, indent=4)
print("✅ Training history saved to models/training_history_simplecnn.json")
print("="*60)
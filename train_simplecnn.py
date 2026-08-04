import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os, json


dataset_path = "data/dataset_makanan_indonesia"
model_save_path = "models/food_model.h5"
class_indices_path = "models/class_indices.json"


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

# Load Data Latih (80%)
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load Data Validasi (20%)
val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# Terdiri dari 3 Convolution Layer + Max Pooling
model = Sequential([
    
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
   
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation='softmax') # 18 Kelas
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n🚀 Memulai Training SimpleCNN (Asli dari Nol)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50, # Memberi kesempatan model belajar lebih lama
    callbacks=[early_stop]
)


os.makedirs("models", exist_ok=True)
model.save(model_save_path)
print(f"\n✅ Model SimpleCNN asli berhasil disimpan di: {model_save_path}")


with open(class_indices_path, "w") as f:
    json.dump(train_gen.class_indices, f)
print(f"✅ Mapping kelas disimpan di: {class_indices_path}")


with open("models/training_history_simplecnn.json", "w") as f:
    json.dump(history.history, f, indent=4)
print("✅ Training history saved to models/training_history_simplecnn.json")

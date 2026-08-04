import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os, json


dataset_path = "data/dataset_makanan_indonesia"


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
    target_size=(224, 224),  # EfficientNet optimal di 224x224
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
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

assert len(train_gen.class_indices) == 18, f"❌ ERROR: Expected 18 classes, got {len(train_gen.class_indices)}"


base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze di tahap 1


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation='softmax')
])


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

print("🚀 Training tahap 1 (transfer learning)...")
history_stage1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)


print("🔧 Fine-tuning tahap 2 (unfreeze 50 layer terakhir)...")
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Freeze semua kecuali 50 layer terakhir
    layer.trainable = False


model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_stage2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)


os.makedirs("models", exist_ok=True)
model.save("models/food_model_efficientnet_finetuned.h5")


with open("models/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("✅ Model saved to models/food_model_efficientnet_finetuned.h5")
print("✅ Class indices saved to models/class_indices.json")

print("\n" + "="*60)
print("Menyimpan training history...")

combined_history = {
    'accuracy': history_stage1.history['accuracy'] + history_stage2.history['accuracy'],
    'val_accuracy': history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy'],
    'loss': history_stage1.history['loss'] + history_stage2.history['loss'],
    'val_loss': history_stage1.history['val_loss'] + history_stage2.history['val_loss']
}

with open("models/training_history_efficientnet.json", "w") as f:
    json.dump(combined_history, f, indent=4)
print("✅ Training history saved to models/training_history_efficientnet.json")
print("="*60)
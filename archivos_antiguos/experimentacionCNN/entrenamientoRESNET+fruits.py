import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

DATASET_BASE = "C:\\ARIS\\descargas\\fruits-360_100x100\\fruits-360\\"      
TRAIN_DIR = os.path.join(DATASET_BASE, "training")
TEST_DIR = os.path.join(DATASET_BASE, "test")

MODEL_SAVE_PATH = "modelos\\resnet50_frutas.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 1 # debido a limitacion del portatil
LEARNING_RATE = 0.0001

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Carpeta 'train' no encontrada: {TRAIN_DIR}")
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Carpeta 'test' no encontrada: {TEST_DIR}")

# === Generadores SIN aumento de datos ===
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Clases detectadas:", train_gen.class_indices)
print(f"Número de clases: {train_gen.num_classes}")

# === Modelo: ResNet50 base ===
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

base_model.trainable = False 
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Entrenamiento ===
print("\nEntrenando ResNet50 con fruits360...\n")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    verbose=1
)

model.save(MODEL_SAVE_PATH)

with open("modelos\\clases_frutas_resnet.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print(f"\nModelo guardado: {MODEL_SAVE_PATH}")
print("Archivo de clases guardado: modelos\\clases_frutas_resnet.json")

test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\nPrecisión final en test: {test_acc:.4f} ({test_acc * 100:.2f}%)")
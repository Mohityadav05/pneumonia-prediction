"""
Pneumonia Detection — Training Script
Based on paultimothymooney/detecting-pneumonia-in-x-ray-images (Kaggle),
modernized: fixed deprecated sklearn/imbalanced-learn calls, uses class_weight
instead of undersampling (keeps full dataset -> better accuracy), VGG16 transfer
learning with a fine-tuning stage.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Run this in Kaggle/Colab with the dataset attached, or locally after downloading it.
"""

import os
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---- Config ----
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS_HEAD = 10        # train only the new classifier head
EPOCHS_FINE_TUNE = 8    # unfreeze top VGG16 blocks and fine-tune
DATA_DIR = "chest_xray"  # expects DATA_DIR/train, DATA_DIR/val, DATA_DIR/test
MODEL_OUT = "models/pneumonia_model.h5"

os.makedirs("models", exist_ok=True)

# ---- Data generators ----
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["NORMAL", "PNEUMONIA"],  # ensures label 0=NORMAL, 1=PNEUMONIA
)
val_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["NORMAL", "PNEUMONIA"],
)
test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["NORMAL", "PNEUMONIA"],
    shuffle=False,
)

# ---- Class weights (fixed API — modern sklearn requires keyword args) ----
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes,
)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)

# ---- Build model: VGG16 base + custom head ----
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze for stage 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)  # binary: pneumonia probability

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy", "AUC"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True),
]

# ---- Stage 1: train classifier head only ----
print("\n=== Stage 1: training classifier head ===")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ---- Stage 2: unfreeze last VGG16 block and fine-tune at low LR ----
print("\n=== Stage 2: fine-tuning top layers ===")
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy", "AUC"])
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ---- Evaluate ----
print("\n=== Test set evaluation ===")
results = model.evaluate(test_gen)
print(dict(zip(model.metrics_names, results)))

# ---- Save final model ----
model.save(MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")

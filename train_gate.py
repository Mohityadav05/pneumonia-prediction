from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

IMG_SIZE = 150
os.makedirs("models", exist_ok=True)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1)

train_gen = datagen.flow_from_directory(
    "gate_data/train", target_size=(IMG_SIZE, IMG_SIZE), batch_size=32,
    class_mode="binary", classes=["not_xray", "xray"], subset="training"
)
val_gen = datagen.flow_from_directory(
    "gate_data/train", target_size=(IMG_SIZE, IMG_SIZE), batch_size=32,
    class_mode="binary", classes=["not_xray", "xray"], subset="validation"
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
x = Dense(64, activation="relu")(x)
out = Dense(1, activation="sigmoid")(x)

gate_model = Model(base.input, out)
gate_model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
gate_model.fit(train_gen, validation_data=val_gen, epochs=5)
gate_model.save("models/xray_gate.h5")
print("Saved models/xray_gate.h5")
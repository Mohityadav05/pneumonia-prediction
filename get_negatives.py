import os
from tensorflow.keras.datasets import cifar10
from PIL import Image

(x_train, _), _ = cifar10.load_data()
os.makedirs("gate_data/train/not_xray", exist_ok=True)

for i, img_arr in enumerate(x_train[:3000]):
    Image.fromarray(img_arr).save(f"gate_data/train/not_xray/img_{i}.jpg")

print("Saved 3000 non-xray images to gate_data/train/not_xray/")
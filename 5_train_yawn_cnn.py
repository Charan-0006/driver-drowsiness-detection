# 5_train_yawn_cnn.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt

# FIXED: Correct dataset path
DATA_ROOT = "datasets_cnn/mouth"   # NOT datasets_cnn/yawn
IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 8
WEIGHTS_OUT = "weights/yawn_model.h5"
Path("weights").mkdir(exist_ok=True)

def build_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE,3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(base.input, out)
    m.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def train():
    train_path = Path(DATA_ROOT) / "train"
    val_path   = Path(DATA_ROOT) / "val"

    # Data generators
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    ).flow_from_directory(
        train_path, 
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        val_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # ------- CLASS WEIGHTS (IMPORTANT) -------
    class_indices = train_gen.class_indices  # {'no_yawn':0, 'yawn':1}
    counts = {cls: len(list((train_path/cls).glob("*.jpg"))) for cls in class_indices}

    y = []
    for cls, idx in class_indices.items():
        y += [idx] * counts[cls]

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=np.array(y)
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights_array)}

    print("Class Indices:", class_indices)
    print("Class Weights:", class_weights)
    print("Train Counts:", counts)

    # -----------------------------------------

    model = build_model()

    ck = ModelCheckpoint(WEIGHTS_OUT, save_best_only=True,
                         monitor='val_accuracy', mode='max')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[ck, es]
    )

    print("Saved model to:", WEIGHTS_OUT)
def plot_train_history(history, title="Model"):
    plt.figure(figsize=(12,5))

    # Accuracy Plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{title} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    train()

 # 6_train_head_cnn.py
# Head pose binary classifier (normal vs tilt)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt

DATA_ROOT = "datasets_cnn/head"
IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 8
WEIGHTS_OUT = "weights/head_model.h5"
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
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1, horizontal_flip=True).flow_from_directory(Path(DATA_ROOT)/"train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(Path(DATA_ROOT)/"val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
    model = build_model()
    ck = ModelCheckpoint(WEIGHTS_OUT, save_best_only=True, monitor='val_accuracy', mode='max')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[ck,es])
    print("Saved", WEIGHTS_OUT)




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

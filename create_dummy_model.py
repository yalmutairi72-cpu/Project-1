"""
Generates a dummy keras_model.h5 + labels.txt that match the exact
Teachable Machine output format. Replace both files once you export
your real model from Teachable Machine.
"""
import numpy as np
import tensorflow as tf

# Must match what your real Teachable Machine model will output
CLASSES = ["Background", "Alpha", "Beta", "Gamma"]

# Teachable Machine uses MobileNetV2 224x224 input, softmax output
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None,
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
model = tf.keras.Model(inputs=base.input, outputs=x)
model.compile(optimizer="adam", loss="categorical_crossentropy")

model.save("keras_model.h5")
print(f"Saved keras_model.h5  ({model.count_params():,} params)")

with open("labels.txt", "w") as f:
    for i, name in enumerate(CLASSES):
        f.write(f"{i} {name}\n")
print("Saved labels.txt:", CLASSES)

# U-Net Image Segmentation (TensorFlow / Keras)

A minimal, readable implementation of **U-Net** for image segmentation using `tf.keras`.  
The model performs **binary segmentation** (e.g., object vs. background) by default and can be extended to multi-class tasks.

---

## ✨ What’s inside

- Classic **U-Net**: encoder–decoder with **skip connections**
- **Conv–Conv–MaxPool** down the encoder, **Conv2DTranspose + concat** up the decoder
- Input normalization layer and light **Dropout** for regularization
- Output: 1-channel mask with **sigmoid** (binary segmentation)
- Ready to compile & train with your dataset

---

## 🧠 U-Net in one paragraph

U-Net learns to **color every pixel** with the correct class.  
- The **encoder** (left) uses `Conv2D → Conv2D → MaxPool` blocks to find features (edges, textures, shapes) and **shrink** the image so it can understand the big picture.  
- The **decoder** (right) uses `Conv2DTranspose` to **upsample** (make the image big again) and **concatenates** encoder features (skip connections) to recover sharp boundaries.  
- The final `Conv2D(1, kernel=1, activation="sigmoid")` outputs a binary mask the same size as the input.

---

## 🧱 Model code (core)

> Put this into `model_unet.py` or your notebook. Uses only `tf.keras.*`.

```python
import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# BUILD U-NET
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

# ----- Encoder
c1 = tf.keras.layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D(2)(c1)

c2 = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D(2)(c2)

c3 = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D(2)(c3)

c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(2)(c4)

# Bottleneck
c5 = tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# ----- Decoder
u6 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
u6 = tf.keras.layers.Concatenate()([u6, c4])
c6 = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.1)(c6)
c6 = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
u7 = tf.keras.layers.Concatenate()([u7, c3])
c7 = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.1)(c7)
c7 = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
u8 = tf.keras.layers.Concatenate()([u8, c2])
c8 = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
u9 = tf.keras.layers.Concatenate()([u9, c1])
c9 = tf.keras.layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c9)

# Output (binary mask)
outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

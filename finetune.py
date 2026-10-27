#this file is dedicated to finetuning the model

#imports

import tensorflow as tf
from keras import layers

from data import train_ds, val_ds, weights
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



model = tf.keras.models.load_model("best.keras")
aug = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
base = model.get_layer('efficientnetv2-b1')

inputs = tf.keras.Input(shape=(240, 240, 3))
x = aug(inputs)
x = base(x, training = False)
x = model.get_layer('global_average_pooling2d')(x)
x = model.get_layer('dropout')(x)
outputs = model.get_layer('dense')(x)
model = tf.keras.Model(inputs, outputs)

base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

for layer in base.layers[-30:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=15,
          class_weight=weights,
          callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
          ModelCheckpoint("best_ft_aug.keras", monitor='val_loss', save_best_only=True)])

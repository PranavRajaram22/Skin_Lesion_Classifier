#this file is dedicated to finetuning the model

#imports

import tensorflow as tf
from data import train_ds, val_ds, weights
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



model = tf.keras.models.load_model("best.keras")
base = model.get_layer('efficientnetv2-b1')
base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

for layer in base.layers[-30:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=15,
          class_weight=weights,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
          ModelCheckpoint("best_ft.keras", monitor='val_loss', save_best_only=True)])

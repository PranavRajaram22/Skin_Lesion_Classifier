#imports
from data import train_ds, val_ds, weights
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#time to actually build the model!

#creating the base of the model, taking it from EfficientNetV2B1. excluding the top because ill add my own 7-class dense head
#reloading pretrained weights by tensorflow, and passing the input shape.
#freezing this layers because ill be training the dense part first and dont want to touch the training already done on V2B1 yet.
model_base = tf.keras.applications.EfficientNetV2B1(include_top=False, weights="imagenet", input_shape=(240,240,3))
model_base.trainable = False

#building the input layers
inputs = tf.keras.Input(shape=(240, 240,3))
x = model_base(inputs, training=False) #basically says that the output will have shape (None, 8,8, 1280).
x = tf.keras.layers.GlobalAveragePooling2D()(x) #creates the pooling average (averages each 8x8 values into 1 number), feeds x in, then passes it to x. shape is now (None, 1280)
x = tf.keras.layers.Dropout(0.2)(x) #to reduce overfitting. forces 20% of the 1280 values to 0, and scale the rest up accordingly. only during training. dont want the model to rely solely one 1 feature for classification

#building the dense layer as the output. using softmax to get readable probabilities that an image is 1 of 7 classes.
outputs = tf.keras.layers.Dense(7, activation='softmax')(x)

#putting the pieces together
model = tf.keras.Model(inputs, outputs)


#compiling the model with its required settings.
#optimizer is Adam bc i want to improve gradient descent.
#sparce categorical crossentropy bc the labels are integers (0, 1, 2....)
#reporting accuracy
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#train upto 15 epochs, stopping if val_loss doesnt improve after 3 epochs
history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=15,
          class_weight=weights,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
          ModelCheckpoint("best.keras", monitor='val_loss', save_best_only=True)])



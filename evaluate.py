#imports

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report
from data import val_ds, val, classes

#reloading fine_tuned model
model = tf.keras.models.load_model('best_ft.keras')

#predict probabilities
probs = model.predict(val_ds)
pred = probs.argmax(axis=1)
true = val['label'].to_numpy()

print(classification_report(true, pred, target_names=classes))
cm = confusion_matrix(true, pred, labels=range(7))
print(cm)



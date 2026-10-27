#imports

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from data import val_ds, val, test_ds, test, classes

#reloading augmented model
model = tf.keras.models.load_model('best_ft_aug.keras')

#predict probabilities

probs = model.predict(test_ds)
pred = probs.argmax(axis=1)
true = test['label'].to_numpy()

print(classification_report(true, pred, target_names=classes))
cm = confusion_matrix(true, pred, labels=range(7))
print(cm)

ras = roc_auc_score(true==classes.index('mel'), probs[:, classes.index('mel')])

print(ras)

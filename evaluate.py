#imports

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data import val_ds, val, classes, test_ds, test
import pandas as pd
import matplotlib.pyplot as plt

#reloading model from best.keras
model = tf.keras.models.load_model('best.keras')

#predict probabilities
probs = model.predict(test_ds)
pred = probs.argmax(axis=1)
true = test['label'].to_numpy()

cf = classification_report(true, pred, target_names=classes, output_dict=True)
df = pd.DataFrame(cf).T.round(2)
df['support'] = df['support'].astype(int)
print(df.to_markdown())

cm = confusion_matrix(true, pred, labels=range(7), normalize='true')
cmd = ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap='Blues', values_format='.2f')
plt.title("Normalized Baseline Confusion Matrix")
plt.savefig('baseline_confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
plt.close()

cm = confusion_matrix(true, pred, labels=range(7))
cmd = ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap='Blues', values_format='d')
plt.title("Baseline Confusion Matrix")
plt.savefig('baseline_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
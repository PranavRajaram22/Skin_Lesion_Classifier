#imports

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from data import val_ds, val, test_ds, test, classes
import numpy as np

#reloading augmented model

model = tf.keras.models.load_model('best_ft2.keras')

#predict probabilities

probs = model.predict(test_ds)
pred = probs.argmax(axis=1)
true = test['label'].to_numpy()



import numpy as np
m = classes.index('mel'); n = classes.index('nv')
pm = probs[:, m]

def pick(mask, highest=True, k=3):
    idx = np.where(mask)[0]
    return idx[np.argsort(pm[idx])[::-1 if highest else 1][:k]]

correct_mel = pick((true == m) & (pred == m))
missed_mel  = pick((true == m) & (pred == n), highest=False)
false_mel   = pick((true == n) & (pred == m))
rng = np.random.default_rng(0)
random_nv   = rng.choice(np.where((true == n) & (pred == n))[0], 3, replace=False)

for name, group in [('correct_mel', correct_mel), ('missed_mel', missed_mel),
                    ('false_mel', false_mel), ('random_nv', random_nv)]:
    print(name, test.iloc[group]['path'].tolist(), pm[group].round(2))


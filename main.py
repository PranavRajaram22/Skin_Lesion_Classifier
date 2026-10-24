#imports

import pandas as pd
import pathlib
from sklearn.model_selection import StratifiedGroupKFold

#just to make opening folders easier
root = pathlib.Path("skin-cancer-mnist-ham10000/versions/2")

#opening the dataset and storing it in the dataset df using pandas
df = pd.read_csv(root /  "HAM10000_metadata.csv")

#getting the names of each class and associating it with a number using a dict
classes = sorted(df["dx"].unique())
label_of = {c: i for i, c in enumerate(classes)}

#mapping each image's name to the number established in label_of
df['label'] = df['dx'].map(label_of)


paths = {p.stem: p for p in root.glob("HAM10000_images_part_*/*.jpg")}

df['path'] = df['image_id'].map(paths)


X = df.drop(columns='label')
y = df['label']

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

folds = list(sgkf.split(X, y, groups = df['lesion_id']))

train_val_idx, test_idx = folds[0]

test = df.iloc[test_idx]
train_val = df.iloc[train_val_idx].reset_index(drop=True)

folds = list(StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42).split(train_val, train_val['label'], groups=train_val['lesion_id']))
train_idx, val_idx = folds[0]

train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]

print(len(train))
print(train['dx'].value_counts())

print(len(test))
print(test['dx'].value_counts())

print(len(val))
print(val['dx'].value_counts())

#length of train split is 6676
#length of test split is 2004
#length of validation split is 1335

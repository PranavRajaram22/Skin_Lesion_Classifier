#imports

import pandas as pd
import pathlib
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf

#just to make opening folders easier
root = pathlib.Path("skin-cancer-mnist-ham10000/versions/2")

#opening the dataset and storing it in the dataset df using pandas
df = pd.read_csv(root /  "HAM10000_metadata.csv")

#getting the names of each class and associating it with a number using a dict
classes = sorted(df["dx"].unique())
label_of = {c: i for i, c in enumerate(classes)}

#mapping each image's class name to the number established in label_of,
#and putting each number as a label, a new column on the dataframe
df['label'] = df['dx'].map(label_of)

#putting another column in the dataframe for paths to individual images.
#helpful because for some reason HAM10000 decided to put the images in 2 files
paths = {p.stem: p for p in root.glob("HAM10000_images_part_*/*.jpg")}
df['path'] = df['image_id'].map(paths)

#the next several lines are dedicated to getting a train/test/val split

#Splitting between input and output. the model will only read images from X['path']
X = df.drop(columns='label')
y = df['label']

#essentially creating split "settings"
#lesion-grouped 5-fold splitter
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

#all photos of 1 lesion need to be in the same group. nearly identical photos shouldnt be in the train and test splits.
folds = list(sgkf.split(X, y, groups = df['lesion_id']))

#picking the first fold as the choice for training/validation and test split. just getting row numbers for now
#also cant split into all 3 at once bc skgf doesnt work like that, so will split train and val soon
train_val_idx, test_idx = folds[0]

#converting row numbers into actual data for train/val and test. no need to drop indices for test as we arent splitting it again
test = df.iloc[test_idx]
train_val = df.iloc[train_val_idx].reset_index(drop=True)

#now doing the exact same thing again, but specifically for splitting training and validation
folds = list(StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42).split(train_val, train_val['label'], groups=train_val['lesion_id']))
train_idx, val_idx = folds[0]


#converting row numbers into actual rows of data again
train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]

#length of train split is 6676
#length of test split is 2004
#length of validation split is 1335

#class weights for training:
weights = dict()

#using the formula for weights because HAM10000 is incredibly imbalanced. rare classes get large weights
#specific to the train split only
for cls in train['label']:
    weight = len(train)/(len(classes)*(train['label'] == cls).sum())
    weights[cls] = weight

print(weights)
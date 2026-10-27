7-class skin lesion classifier on the HAM10000 dataset, with transfer-learning from EfficientNetV2-B1.

Note: This model was created solely as a research project for my portfolio. Please do not use this as a clinical or diagnostic tool under any circumstances. 


Sources:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data [1]

https://my.clevelandclinic.org/health/diseases/24296-skin-lesions [2]

https://www.codetrade.io/blog/grad-cam-a-complete-guide-with-example/ [3]

https://arxiv.org/abs/1610.02391 [4]

https://challenge.isic-archive.com/landing/2018/47/ [5] 


**About the data and dataset**

The dataset used was the HAM10000 dataset [1]. It consists of 10015 close-up images of skin lesions, and each skin lesion belongs to 1 of 7 classes. A skin lesion is defined as an area on the skin that is different from the skin around it. [2] Most are not cancerous and harmless (benign), but some are cancerous, or are a symptom of infections and autoimmune diseases. The 7 classes along with their acronyms are listed below. Note that acronyms will be the naming scheme for this ReadMe.


1. Melanoma (mel) | Count in dataset: 1113
2. Melanocytic Nevus (nv) | Count in dataset: 6705
3. Basal Cell Carcinoma (bcc) | Count in dataset: 514
4. Actinic Keratosis (akiec) | Count in dataset: 327
5. Benign Keratosis (bkl) | Count in dataset: 1099
6. Dermatofibroma (df) | Count in dataset: 115
7. Vascular lesion (vasc) | Count in dataset: 142


<img width="1874" height="960" alt="image" src="https://github.com/user-attachments/assets/f8cdd06d-8efb-4c77-883f-67b18c5c6f32" />
[5]

**The Method**

First, the images were downloaded from Kaggle directly [1] into the project folder using code in download_files.py. Then, data was processed and grouped into viable training, validation, and testing splits. This was done using SciKitLearn's StratifiedGroupKFold function. This function creates subsets of data, known as 'folds', and ensures that 1. all of the photos of 1 lesion are in the same fold and 2. that the distribution of classes in each fold roughly matches the original distribution of the classes in the dataset. This ensures that near-duplicate images of the same photo never appear in both training and testing data, which would inflate accuracy. This function was used twice, along with other functions, to create 3 splits of data; training data, validation data, testing data. 

Next, the model was created. The model is based off of EfficientNetV2B1 from TensorFlow, pretrained on images from ImageNet. Inputs were taken in as 240 pixels by 240 pixels, with 3 color channels. After EfficientNetV2B1, GlobalAveragePooling and DropOut by 20% were added. The EfficientNet layers were frozen for the first round of training. Next, a dense head with 7 neurons and a softmax activation was added. It is this head that gets trained. The other layers were frozen to stop the training from affecting both the untrained dense head and the ImageNet training done on EfficientNetV2B1. 

Finally, the model was trained on the images, with an Adam optimizer at a learning rate of 10^-3, and sparse categorical crossentropy as the loss because the classes are encoded as integers. Furthermore, early stopping with a patience of 3 and model checkpoints were enabled. 

After training, evaluate.py was used on the model (best.keras) using the test data. As implied, this model has no finetuning nor data augmentation involved. Below are the classification report and the confusion matrices (regular and normalized)


|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| akiec        |        0.25 |     0.57 |       0.35 |        44 |
| bcc          |        0.43 |     0.52 |       0.47 |        69 |
| bkl          |        0.57 |     0.55 |       0.56 |       146 |
| df           |        0.2  |     0.73 |       0.31 |        15 |
| mel          |        0.39 |     0.6  |       0.47 |       148 |
| nv           |        0.96 |     0.74 |       0.84 |       894 |
| vasc         |        0.49 |     0.89 |       0.63 |        19 |
| accuracy     |        0.69 |     0.69 |       0.69 |         0 |
| macro avg    |        0.47 |     0.66 |       0.52 |      1335 |
| weighted avg |        0.79 |     0.69 |       0.72 |      1335 |


<img width="815" height="680" alt="image" src="https://github.com/user-attachments/assets/dce8df9e-fd71-4d91-8b54-85a24eb6b454" />
<img width="796" height="680" alt="image" src="https://github.com/user-attachments/assets/e763f00a-f208-4d62-8559-df471e6d38c1" />


7-class skin lesion classifier on the HAM10000 dataset, with transfer-learning from EfficientNetV2-B1.

Note: This model was created solely as a research project for my portfolio. Please do not use this as a clinical or diagnostic tool under any circumstances. 


Sources:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data [1]

https://my.clevelandclinic.org/health/diseases/24296-skin-lesions [2]

https://www.codetrade.io/blog/grad-cam-a-complete-guide-with-example/ [3]

https://arxiv.org/abs/1610.02391 [4]

https://challenge.isic-archive.com/landing/2018/47/ [5] 

https://en.wikipedia.org/wiki/Precision_and_recall [6]

https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884 [7]


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

First, the images were downloaded from Kaggle directly [1] into the project folder using code in download_files.py. Then, data was processed and grouped into viable training, validation, and testing splits. This was done using scikit-learn's StratifiedGroupKFold function. This function creates subsets of data, known as 'folds', and ensures that 1. all of the photos of 1 lesion are in the same fold and 2. that the distribution of classes in each fold roughly matches the original distribution of the classes in the dataset. This ensures that prevents photos of the same lesion from appearing in both training and testing data. This function was used twice, along with other functions, to create 3 splits of data; training data, validation data, testing data. 

Next, the model was created. The model is based off of EfficientNetV2B1 from TensorFlow, pretrained on images from ImageNet. Inputs were taken in as 240 pixels by 240 pixels, with 3 color channels. After EfficientNetV2B1, GlobalAveragePooling and DropOut by 20% were added. The EfficientNet layers were frozen for the first round of training. Next, a dense head with 7 neurons and a softmax activation was added. It is this head that gets trained. The EfficientNet layers were frozen so that the large early gradients from the randomly initialized head could not damage the pretrained ImageNet features.

Finally, the model was trained on the images, with an Adam optimizer at a learning rate of 10^-3, and sparse categorical crossentropy as the loss because the classes are encoded as integers. Furthermore, early stopping with a patience of 3 and model checkpoints were enabled. 

After training, evaluate.py was used on the model (best.keras) using the test data (2004 images). As implied, this model has no finetuning nor data augmentation involved. Below are the classification report and the confusion matrices (regular and normalized)


|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| akiec        |        0.33 |     0.77 |       0.46 |        66 |
| bcc          |        0.41 |     0.47 |       0.44 |       103 |
| bkl          |        0.58 |     0.52 |       0.55 |       220 |
| df           |        0.16 |     0.65 |       0.25 |        23 |
| mel          |        0.39 |     0.52 |       0.45 |       222 |
| nv           |        0.96 |     0.78 |       0.86 |      1341 |
| vasc         |        0.38 |     0.69 |       0.49 |        29 |
| accuracy     |        0.7  |     0.7  |       0.7  |      2004 |
| macro avg    |        0.46 |     0.63 |       0.5  |      2004 |
| weighted avg |        0.79 |     0.7  |       0.73 |      2004 |


<img width="815" height="680" alt="image" src="https://github.com/user-attachments/assets/56bb15ab-28d4-421f-908e-947135b75ffc" />
<img width="796" height="680" alt="image" src="https://github.com/user-attachments/assets/c7f2cfd1-abc3-48ac-9f02-37c3cfcc3ea2" />


The accuracy row in the above table is not informative on its own. This is because guessing nv for every single image already gives an accuracy of 0.67 (1341/2004). 
The most relevant numbers are:

- Macro avg of the F1 score (0.50): This number indicates that the model is okay on overall reliability. It is much better than random guessing (0.14), but low scores on df (0.25) and bcc (0.44) bring the model's reliability down.
- Macro-recall (0.63): This number indicates that the model, on average, correctly identifies 63% of images in each class.
- Macro-precision (0.46): This indicates that the model's flags are correct less than half the time. This shows that the model over-flags: it raises many false alarms.
- mel recall (0.52): This indicates that the model misses 5 out of every 10 images of mel [6]. Not safe for diagnosis, as melanoma is deadly. [7]
- mel precision (0.39): This indicates that about 6 of every 10 images the model flags as mel are not mel [6]. Most of these come from identifying nv as mel, as the 2 look similar. This similarity will be explored using GradCAM later.
- vasc recall (0.69): This indicates that the model correctly identifies images as vasc almost 7 out of 10 times, even with such few images for testing (29). This may be because vasc looks very distinct compared to the other lesions.


Overall, the model, with basic training, has uneven performance. It is most reliable on nv, the most common class (F1 0.86), and weakest where lesions look alike or the sample is small, e.g. mel/nv confusion and df (F1 0.25, from only 23 test images).

**Data augmentation, fine-tuning, and final model selection**

Based on the classification report and confusion matrices, the model could use some improvement. The baseline had a macro-F1 of 0.50 and had difficulty distinguishing between mel and nv. 

First, finetuning was implemented. The last 30 layers of EfficientNetV2B1 were unfrozen, and the learning rate dropped to 10^-5. The number of epochs was up to 15, patience for early stopping increased to 5 epochs, and class weights were kept the same. After fine-tuning once, and then raising the learning rate to 10^-4, the Macro F1 increased in the second finetuning. Macro-F1 went from 0.5 in the baseline to 0.52 in the first finetuned model, to 0.56 in the second finetuned model. Since the second finetuned model is ultimately the chosen model, its classification report and confusion matrices will be shown later. Below is the classification report for the first fine-tuned model. 

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| akiec        |        0.4  |     0.64 |       0.49 |        66 |
| bcc          |        0.39 |     0.58 |       0.47 |       103 |
| bkl          |        0.51 |     0.57 |       0.54 |       220 |
| df           |        0.27 |     0.57 |       0.36 |        23 |
| mel          |        0.39 |     0.55 |       0.46 |       222 |
| nv           |        0.96 |     0.78 |       0.86 |      1341 |
| vasc         |        0.38 |     0.62 |       0.47 |        29 |
| accuracy     |        0.71 |     0.71 |       0.71 |      2004 |
| macro avg    |        0.47 |     0.61 |       0.52 |      2004 |
| weighted avg |        0.78 |     0.71 |       0.74 |      2004 |

As stated before, the Macro-F1 went from 0.50 in the baseline model to 0.52 in the first finetuned model. This is a small improvement. Furthermore, while df precision increased from 0.16 to 0.27, akiec recall dropped from 0.77 to 0.64. Again, the analysis on the second finetuned model will be shown later as it is the best out of all of the other models created in this study.  

(start data augmentation analysis)

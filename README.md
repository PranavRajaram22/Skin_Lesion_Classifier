7-class skin lesion classifier on the HAM10000 dataset, with transfer learning from EfficientNetV2B1.

Note: This model was created solely as a research project for my portfolio. Please do not use this as a clinical or diagnostic tool under any circumstances.


Sources:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data [1]

https://my.clevelandclinic.org/health/diseases/24296-skin-lesions [2]

https://www.codetrade.io/blog/grad-cam-a-complete-guide-with-example/ [3]

https://arxiv.org/abs/1610.02391 [4]

https://challenge.isic-archive.com/landing/2018/47/ [5]

https://en.wikipedia.org/wiki/Precision_and_recall [6]

https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884 [7]

https://doi.org/10.1038/sdata.2018.161 [8]

https://arxiv.org/abs/2104.00298 [9]


**About the data and dataset**

The dataset used was the HAM10000 dataset [1][8]. It consists of 10015 dermoscopic (magnified, close-up) images of skin lesions, and each image belongs to 1 of 7 classes. Some lesions were photographed more than once, so there are fewer unique lesions than images. A skin lesion is defined as an area on the skin that is different from the skin around it [2]. Most are not cancerous and harmless (benign), but some are cancerous, or are a symptom of infections and autoimmune diseases. The 7 classes along with their acronyms are listed below. Note that acronyms will be the naming scheme for this README.


1. Melanoma (mel) | Count in dataset: 1113
2. Melanocytic Nevus (nv) | Count in dataset: 6705
3. Basal Cell Carcinoma (bcc) | Count in dataset: 514
4. Actinic Keratosis / Intraepithelial Carcinoma (akiec) | Count in dataset: 327
5. Benign Keratosis-like Lesion (bkl) | Count in dataset: 1099
6. Dermatofibroma (df) | Count in dataset: 115
7. Vascular Lesion (vasc) | Count in dataset: 142


<img width="1874" height="960" alt="image" src="https://github.com/user-attachments/assets/f8cdd06d-8efb-4c77-883f-67b18c5c6f32" />
[5]

**The Method**

First, the images were downloaded from Kaggle directly [1] into the project folder using code in download_files.py. Then, data was processed and grouped into viable training, validation, and testing splits. This was done using scikit-learn's StratifiedGroupKFold class. This class creates subsets of data, known as "folds", and ensures that 1. all of the photos of 1 lesion are in the same fold and 2. that the distribution of classes in each fold roughly matches the original distribution of the classes in the dataset. This prevents photos of the same lesion from appearing in both training and testing data, which would make the test scores unrealistically high. This class was used twice. First, 1 of 5 folds was held out as testing data. Then, the remaining data was split again into 6 folds, and 1 was held out as validation data. This created 3 splits of data: training data (6676 images, about 67%), validation data (1335 images, about 13%), and testing data (2004 images, about 20%).

Next, the model was created. The model is based on EfficientNetV2B1 [9] from TensorFlow, pretrained on images from ImageNet. Inputs were taken in as 240 pixels by 240 pixels (resized from the original 600 by 450), with 3 color channels. After EfficientNetV2B1, GlobalAveragePooling and Dropout of 20% were added. The EfficientNet layers were frozen for the first round of training. Next, a dense head with 7 neurons and a softmax activation was added. It is this head that gets trained. The EfficientNet layers were frozen so that the large early gradients from the randomly initialized head could not damage the pretrained ImageNet features.

Finally, the model was trained on the images for up to 15 epochs, with an Adam optimizer at a learning rate of 10^-3, and sparse categorical crossentropy as the loss because the classes are encoded as integers. Since the dataset is very imbalanced (nv is 67% of all images), class weights were used so that mistakes on rare classes cost more. The weight of class c is w_c = N / (7 × n_c), where N is the number of training images and n_c is the number of training images in class c. For example, df gets a weight of about 12, while nv gets about 0.2. Furthermore, early stopping with a patience of 3 and model checkpoints were enabled, both monitoring the validation loss.

After training, evaluate.py was used on the model (best.keras) using the validation data (1335 images). As implied, this model has neither fine-tuning nor data augmentation involved. Below are the classification report and the confusion matrices (regular and normalized).

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| akiec        |        0.25 |     0.57 |       0.35 |        44 |
| bcc          |        0.43 |     0.52 |       0.47 |        69 |
| bkl          |        0.57 |     0.55 |       0.56 |       146 |
| df           |        0.2  |     0.73 |       0.31 |        15 |
| mel          |        0.39 |     0.6  |       0.47 |       148 |
| nv           |        0.96 |     0.74 |       0.84 |       894 |
| vasc         |        0.49 |     0.89 |       0.63 |        19 |
| accuracy     |        0.69 |     0.69 |       0.69 |      1335 |
| macro avg    |        0.47 |     0.66 |       0.52 |      1335 |
| weighted avg |        0.79 |     0.69 |       0.72 |      1335 |


<img width="802" height="680" alt="image" src="https://github.com/user-attachments/assets/ae442904-71be-41c9-a842-aafcf4101815" />
<img width="796" height="680" alt="image" src="https://github.com/user-attachments/assets/6c9e2832-c9b6-4014-b31e-891e7be634c1" />

The accuracy row in the above table is not informative on its own. This is because guessing nv for every single image already gives an accuracy of 0.67 (894/1335).
The most relevant numbers are:

- Macro-F1, the macro avg of the F1 score (0.52): Macro-F1 averages the F1 scores of all 7 classes equally, so rare classes count as much as nv. This number indicates that the model is okay on overall reliability. It is much better than random guessing in proportion to class frequencies (0.14), but low scores on df (0.31) and akiec (0.35) bring the model's reliability down.
- Macro-recall (0.66): This number indicates that, averaged over the 7 classes, the model correctly identifies 66% of the images in a class.
- Macro-precision (0.47): This indicates that, averaged over the 7 classes, the model's predictions for a class are correct less than half the time. This shows that the model over-flags: it raises many false alarms. This is an expected side effect of the class weights, which push the model to predict rare classes more often, trading precision for recall.
- mel recall (0.6): This indicates that the model misses about 4 out of every 10 images of mel [6]. This is not safe for diagnosis, as melanoma can be deadly [7].
- mel precision (0.39): This indicates that about 6 of every 10 images the model flags as mel are not mel [6]. Most of these come from identifying nv as mel, as the 2 look similar. The confusion matrices support this. Of the 138 false mel predictions, 111 (80%) were nv and 21 (15%) were bkl. This similarity will be explored using Grad-CAM later.
- vasc recall (0.89): This indicates that the model correctly identifies almost 9 out of 10 vasc images, even though there are so few vasc images for training. This may be because vasc looks very distinct compared to the other lesions. However, with only 19 vasc validation images, 1 image changes vasc recall by about 5 points (1/19 is approximately equal to 0.053), so this number is noisy.

Overall, the model, with basic training, has uneven performance. It is most reliable on nv, the most common class (F1 0.84), and weakest where lesions look alike or there are few training images, e.g. mel/nv confusion and df (F1 0.31, from only 15 validation images, so this score is also noisy).


**Data augmentation, fine-tuning, and final model selection**

Based on the classification report and confusion matrices, the model could use some improvement. The baseline had a Macro-F1 of 0.52 and had difficulty distinguishing between mel and nv.

First, fine-tuning was implemented. The last 30 layers of EfficientNetV2B1 were unfrozen, except for BatchNormalization layers, which were kept frozen so their pretrained ImageNet statistics would not change. The learning rate dropped to 10^-5. Training ran for up to 15 epochs, patience for early stopping increased to 5 epochs, and class weights were kept the same. Macro-F1 went unchanged from the baseline model to the finetuned model. Below is the classification report for the fine-tuned model, again evaluated on the validation data. 

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| akiec        |        0.27 |     0.45 |       0.34 |        44 |
| bcc          |        0.43 |     0.58 |       0.49 |        69 |
| bkl          |        0.52 |     0.67 |       0.59 |       146 |
| df           |        0.23 |     0.67 |       0.34 |        15 |
| mel          |        0.41 |     0.58 |       0.48 |       148 |
| nv           |        0.96 |     0.74 |       0.83 |       894 |
| vasc         |        0.47 |     0.79 |       0.59 |        19 |
| accuracy     |        0.7  |     0.7  |       0.7  |      1335 |
| macro avg    |        0.47 |     0.64 |       0.52 |      1335 |
| weighted avg |        0.78 |     0.7  |       0.72 |      1335 |

As stated before, the Macro-F1 went unchanged from the baseline model to the finetuned model. Furthermore, while df precision increased from 0.2 to 0.23, akiec recall dropped from 0.57 to 0.45. 

No noticeable improvement occurred due to finetuning, so data augmentation was added to finetune.py in hopes of increasing the Macro-F1. This would be helpful as some classes had only a few images in the training data, and randomly varying every training image gives those classes more variety, which would presumably increase their F1 scores, and therefore the Macro-F1. Random horizontal and vertical flips were applied to each training image, then random rotations of up to ±36° (0.1 × 2π rad = 0.2π rad), and then random zooms in or out of up to 10%. These layers are only active during training, so validation and test images were not changed. Below is the classification report of the augmented model, again evaluated on validation data. 

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| akiec        |        0.3  |     0.32 |       0.31 |        44 |
| bcc          |        0.53 |     0.29 |       0.37 |        69 |
| bkl          |        0.43 |     0.62 |       0.51 |       146 |
| df           |        0.1  |     0.8  |       0.17 |        15 |
| mel          |        0.31 |     0.59 |       0.4  |       148 |
| nv           |        0.95 |     0.63 |       0.76 |       894 |
| vasc         |        0.36 |     0.84 |       0.5  |        19 |
| accuracy     |        0.6  |     0.6  |       0.6  |      1335 |
| macro avg    |        0.43 |     0.58 |       0.43 |      1335 |
| weighted avg |        0.76 |     0.6  |       0.65 |      1335 |

The classification report shows that the macro-F1 score actually went down 9 points, from 0.52 to 0.43.

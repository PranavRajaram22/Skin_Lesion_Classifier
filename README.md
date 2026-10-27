7-Class skin lesion classifier on the HAM10000 dataset, with transfer-learning from EfficientNetV2-B1.

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

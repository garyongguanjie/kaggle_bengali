# Baseline Models
Baseline Resnet18 Notebook:
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HRBWzEE009s9KySGiDEq1RCSGHcYx2z6)\
We are using Pytorch's implementation of Resnet18 as a baseline. For VGG, AlexNet, GoogleNet and MobileNet, just change the library name to the corresponding model names in the notebook.

# Model 1 (Longer Tails, sSE Pooling and SE-ResneXt Block)
Resnet34 longer tails Training notebook see `resnet34_ensemble_random_erase.ipynb`\
Training Notebook 30-epoch (Kaggle Kernel) [se-resnext50 training code](https://www.kaggle.com/garyongguanjie/seresnext-50-train-public)\
Inference Notebook 30-epoch (Kaggle Kernel) [se-resnext50](https://www.kaggle.com/garyongguanjie/seresnext-50-inference)\
Training full 105 epoch snapshot ensemble see `seresnext_ensemble.ipynb`

# Model 2 (Seen and Unseen Model)
TODO

# Download Datasets
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EVrOU5p0-FhBggIuqB-rsCgBVzRTExFWLEjXdAVDwa1AQQ?e=FoLgld&download=1" -O train_full.zip
```
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EQGuIT9UUFlDv4IVasLW2dIBANJ5TPjK_hJfZ4yZS11LJQ?e=k6chXZ&download=1" -O split-0.zip
```
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EbxFBne-jWFMuP1dYk6H_DgBV5IJLQPt2BYtNW8dv0PNew?e=761f5d&download=1" -O unseen-val.zip
```
```
unzip <filename>
```

# Data Augmentation Colab Visualisation
This uses the albumentation library -> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-qal9-GSX54R3Z0ZbZKGfS0b4k8FS1ji)

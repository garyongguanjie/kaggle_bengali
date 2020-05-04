# Kaggle: Bengali Handwritten Grapheme Competition
50.035 Computer Vision Project 2020

## Baseline Models
Baseline Resnet18 Notebook:
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HRBWzEE009s9KySGiDEq1RCSGHcYx2z6)\
We are using Pytorch's implementation of Resnet18 as a baseline. For VGG, AlexNet, GoogleNet and MobileNet, just change the library name to the corresponding model names in the notebook.

## Model 1 (Longer Tails, sSE Pooling and SE-ResneXt Block) 

### Download Datasets
Full Training npy file 200k graphemes
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EVrOU5p0-FhBggIuqB-rsCgBVzRTExFWLEjXdAVDwa1AQQ?e=FoLgld&download=1" -O train_full.zip
```
Seen training set ~77% of 200k and seen validation set ~20% of 200k
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EQGuIT9UUFlDv4IVasLW2dIBANJ5TPjK_hJfZ4yZS11LJQ?e=k6chXZ&download=1" -O split-0.zip
```
Unseen validation set ~3% of 200k
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EbxFBne-jWFMuP1dYk6H_DgBV5IJLQPt2BYtNW8dv0PNew?e=761f5d&download=1" -O unseen-val.zip
```
```
unzip <filename>
```


Resnet34 longer tails Training notebook see `resnet34_ensemble_random_erase.ipynb`\
Training Notebook 30-epoch (Kaggle Kernel) [se-resnext50 training code](https://www.kaggle.com/garyongguanjie/seresnext-50-train?scriptVersionId=33097515)\
Inference Notebook 30-epoch (Kaggle Kernel) [se-resnext50](https://www.kaggle.com/garyongguanjie/seresnext-50-inference)\
Training full 105 epoch snapshot ensemble see `seresnext_ensemble.ipynb`

## Model 2 (Seen and Unseen Model)
### Setup
Download necessay dataset from OneDrive. In order for you to save time training, we also provide the results and trained weights from training.

```shell
# 113M
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/pengfei_hong_mymail_sutd_edu_sg/Ee08eKeH7lRModyxrDZp2_QBGj8m0_23w_9ezMg0WcqdqQ?e=kRgXqE&download=1" -O save.zip
unzip save.zip
mv save seen-unseen

# 3.8G 
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/pengfei_hong_mymail_sutd_edu_sg/EXFxcYpnrDlJjoRrGnh86AcBbe1YlvkKt6wxxVxmioiL_w?e=wTOFhv&download=1" -O bengali.zip
unzip bengali.zip
mv bengali seen-unseen
```

Next, we need to install some required libraries.

1. We assume you already have PyTorch (we used 1.4) and torchvision installed
2. Install other packages with `pip install -r seen-unseen/requirements.txt`
3. You can choose to train faster with apex, [Install Apex following the instructions here](https://github.com/NVIDIA/apex#quick-start)

### Training and Inference
- The training code for seen model is in `seen-unseen/seen-model-bengali.ipynb`, it took 4 hours of training on 2080 Ti.
- The training code for unseen model is in `seen-unseen/unseen-model-bengali.ipynb`, it took 5.5 hours of training on 2080 Ti.
- The Inference code for kaggle submission is in `seen-unseen/inference-seen-unseen-models.ipynb`, it generates the `submission.csv` that can be used for submission on kaggle.


# Data Augmentation Colab Visualisation
This uses the albumentation library -> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-qal9-GSX54R3Z0ZbZKGfS0b4k8FS1ji)

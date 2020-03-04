# kaggle_bengali
# Augmentation Colab visualisation
This uses the albumentation library -> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-qal9-GSX54R3Z0ZbZKGfS0b4k8FS1ji)

# Utilities
Non esential code here \
`bengali_numpy_dump` used to clean data before passing into dataloaders

# Download npy files
Transformation here is used https://www.kaggle.com/iafoss/image-preprocessing-128x128. \
Guide for **kaggle inference dataloader** is here https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-inference. \
Spliting used https://www.kaggle.com/yiheng/iterative-stratification which uses https://github.com/trent-b/iterative-stratification

## File explanation
Train-val-split according to ratio in the file name.
## Full link
https://sutdapac-my.sharepoint.com/:f:/g/personal/gary_ong_mymail_sutd_edu_sg/Eo7SYmp7NttFqveH-3dxY9wBZiaBh4FpWtWe0Xj-zZaWgw

## Download straight to colab
shuffle-80-10-10.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/ETBzVmouTtFBgdOjpNrSqKgBajAK70K7cDil6rCtBAaUew?e=MUu6r4&download=1" -O shuffle-80-10-10.zip
```
shuffle-70-15-15.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/ESIe6K9tanZJrJ5bUlIUBrwBy7RQ57Ul2wp7hBp8AHRk0A?e=NSSs3B&download=1" -O shuffle-70-15-15.zip
```

train_full.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EasIHR_n5-1Egs8_6fT2md4BAGn-3Wa2ywTbiWr_VRuK8g?e=GMagMx&download=1" -O train_full.zip
```

## Unzip files

```
unzip <filename>
```
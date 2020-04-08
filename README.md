# kaggle_bengali
# Guide to not waste Kaggle submissions
Test on the test set inside the kernel itself and make sure everything runs.\
Check if your output matches other people's kernel.\
Test on the train set inside the kernel to make sure everything fits in memory or doenst take too long. \
Make sure to change it back to the test set clear your memory run everything again.\
Click commit than wait for commit to be done click open->output-> submit to competition.
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

## Download straight to colab
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EQGuIT9UUFlDv4IVasLW2dIBANJ5TPjK_hJfZ4yZS11LJQ?e=k6chXZ&download=1" split-0.zip
```
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EbxFBne-jWFMuP1dYk6H_DgBV5IJLQPt2BYtNW8dv0PNew?e=761f5d&download=1" unseen-val.zip
```
```
unzip <filename>
```

# kaggle_bengali
# Utilities
Non esential code here \
`bengali_numpy_dump` used to clean data before passing into dataloaders

# Download npy files
Transformation here is used https://www.kaggle.com/iafoss/image-preprocessing-128x128. \
Guide for **kaggle inference dataloader** is here https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-inference. \
Spliting used https://www.kaggle.com/yiheng/iterative-stratification which uses https://github.com/trent-b/iterative-stratification

## File explanation

You only need to use **one** split file. The rest are just different splits of the same data in case we want to do cross validation. Train_full contains unsplit data.

5split.zip gives data split into 5 chunks. Because cannot fit everything into memory when I tested, gives os 22 error apparently this happens on windows and not on colab. In colab can just dump everything into memory

train.csv -> original file

train_0.csv ->label file that corresponds to input training_set_0.npy

validation_0.csv -> label file that corresponds to input validation_set_0.npy

## Full link
https://sutdapac-my.sharepoint.com/:f:/g/personal/gary_ong_mymail_sutd_edu_sg/Eo7SYmp7NttFqveH-3dxY9wBZiaBh4FpWtWe0Xj-zZaWgw

## Download straight to colab
80-10-10.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EULzJcq-ZopEg25HAZGPaxYBguHDpAafpAFkIr5RxTtqkQ?e=XOHEjG&download=1" -O 80-10-10.zip
```
70-15-15.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EYhOtCLwVqlGtfZHCyuPewEB3za-ySztSTob2ghh1WC9zw?e=7ZvQEg&download=1" -O 70-15-15.zip
```

5split.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EY4SYAbFE99BqSW3A5LqqqUBbA73KjTLeLJ2-Emko4snVw?e=04bPHS&download=1" -O 5split.zip
```
split_0.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EeL3Am93wixKjRBort1fGesBD-j5tadq6qe5ASIpcM-0_g?e=j3NfHH&download=1" -O split_0.zip
```

split_1.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EXuBaqTIkVxAn64x3kfilrAB4Tb9Y-Zptloa5aYR_mrA9Q?e=DoPoVK&download=1" -O split_1.zip
```
split_2.zip

```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/ET7XjumIhF5DplF2f3DswGIBM_oQhKMTkcCyGYdSJUiL9A?e=W4S2TQ&download=1" -O split_2.zip
```
split_3.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EVujg4Sc0JBHhOzXLJQpQFwByRUDz3a-EmYCFbkHuyEWcg?e=hVAKgE&download=1" -O split_3.zip
```
split_4.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EX69cK-TkXZClXckbrAjHdoB6d3HlaWIDsIT0j6kdrykMw?e=CxPFzg&download=1" -O split_4.zip
```
train_full.zip
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EasIHR_n5-1Egs8_6fT2md4BAGn-3Wa2ywTbiWr_VRuK8g?e=GMagMx&download=1" -O train_full.zip
```

## Unzip files

```
unzip <filename>
```
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

## Download straight to colab
Note files regenerated with different preprocessing
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EayDZjbTF3JAi1VTlyiU5skB5VLEdUjjFTrktsTm29WB_Q?e=aO1YJR&download=1" -O 701515.zip
```
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EUVuXRi5kChMvORVE7enxvIBu4l43wk1KclBOiaPHOLLtw?e=6RRFi9" -O 801010.zip
```
## Unzip files

```
unzip <filename>
```
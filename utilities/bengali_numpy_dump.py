#Adapted from https://www.kaggle.com/iafoss/image-preprocessing-128x128
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    HEIGHT = 137
    WIDTH = 236
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

train_parquets = ["bengaliai-cv19/train_image_data_0.parquet","bengaliai-cv19/train_image_data_1.parquet","bengaliai-cv19/train_image_data_2.parquet","bengaliai-cv19/train_image_data_3.parquet"]

start = 0
end = 0
for parq in train_parquets:
    df = pd.read_parquet(parq)
    img_list = []
    for i in tqdm(range(len(df))):
        img = 255 -df.loc[i,:].values[1:].astype(np.uint8).reshape(137,236)
        img = (img*(255.0/img.max())).astype(np.uint8)
        img = crop_resize(img)
        img_list.append(img)
    end = start + len(df) - 1
    np.save("{}-{}".format(start,end),np.array(img_list))
    start = end + 1

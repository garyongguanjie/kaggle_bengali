import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

DIR = "./bengaliai-cv19/"
files = ["train_image_data_0.parquet","train_image_data_1.parquet","train_image_data_2.parquet","train_image_data_3.parquet"]

all_image_list = []
for f in files:
    path = DIR + f
    df = pd.read_parquet(path)
    values = 255 - df.iloc[:, 1:].values.reshape(-1, 137, 236).astype(np.uint8)
    img_list = []
    for i in tqdm(range(len(values))):
        img = cv2.resize(values[i],(128,128))
        img_list.append(img)
    
    img_list = np.array(img_list)
    all_image_list.append(img_list)

all_image_list = np.concatenate(all_image_list)
np.save("train_full_128",all_image_list)
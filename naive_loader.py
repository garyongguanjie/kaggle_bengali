# Naive Loader just dumps everything into memory
# Check performance if performance is slow
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from PIL import Image
class BengaliDataset(Dataset):
    def __init__(self,npy_file,label_csv,transform=None):
        self.npy_file = np.load(npy_file)
        self.transform = transform
        df = pd.read_csv(label_csv)
        # for faster access i think
        self.grapheme_root = df["grapheme_root"].values
        self.vowel_diacritic = df["vowel_diacritic"].values
        self.consonant_diacritic = df["consonant_diacritic"].values


    def __getitem__(self, index):
        image_arr = self.npy_file[index]
        image_arr = Image.fromarray(image_arr).convert("RGB")
        image_arr = self.transform(image_arr)
        grapheme_root = torch.Tensor([self.grapheme_root[index]]).long()
        vowel_diacritic = torch.Tensor([self.vowel_diacritic[index]]).long()
        consonant_diacritic = torch.Tensor([self.consonant_diacritic[index]]).long()
        
        return {"image":image_arr,"grapheme_root":grapheme_root,"vowel_diacritic":vowel_diacritic,"consonant_diacritic":consonant_diacritic}

    def __len__(self):
        return self.npy_file.shape[0]
    
if __name__ == "__main__":
    train_data = BengaliDataset("split_0/training_set_0.npy","split_0/training_0.csv")
    print(len(train_data))
    print(train_data[0])
# Naive Loader just dumps everything into memory
# Check performance if performance is slow
from torch.utils.data import Dataset
import pandas as pd
import torch
class BengaliDataset(Dataset):
    def __init__(self,npy_file,label_csv,transform=None):
        super().__init__(self)
        self.npy_file = npy_file
        df = pd.read_csv(label_csv)
        # for faster access i think
        self.grapheme_root = df["grapheme_root"].values
        self.vowel_diacritic = df["vowel_diacritic"].values
        self.consonant_diacritic = df["consonant_diacritic"].values


    def __getitem__(self, index):
        image_arr = self.npy_file[index]
        image_arr = torch.from_numpy(image_arr).repeat(3,1,1)#convert tp 3 channels so can pass easily to imagenet models
        if self.transform:
            image_arr = self.transform(image_arr)
        grapheme_root = torch.Tensor([self.grapheme_root[index]],dtype=torch.long)
        vowel_diacritic = torch.Tensor([self.vowel_diacritic[index]],dtype=torch.long)
        consonant_diacritic = torch.Tensor([self.consonant_diacritic[index]],dtype=torch.long)
        
        return {"image":image_arr,"grapheme_root":grapheme_root,"vowel_diacritic":vowel_diacritic,"consonant_diacritic":consonant_diacritic}

    def __len__(self):
        return self.npy.shape[0]
    
if __name__ == "__main__":
    train_data = BengaliDataset()
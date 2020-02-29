# Adapted from https://www.kaggle.com/yiheng/iterative-stratification
import numpy as np
import pandas as pd

#get data
nfold = 5
seed = 12

train_df = pd.read_csv('bengaliai-cv19/train.csv')
train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))

X, y = train_df[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]\
.values[:,0], train_df.values[:,1:]

train_df['fold'] = np.nan

#split data
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=seed)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    train_df.iloc[test_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')

x = np.load("train.npy")

for i in range(5):
    index = train_df[train_df['fold']==i].index.values
    tr_set = x[index]
    tr_labels = train_df.loc[index]
    tr_labels.to_csv(f"set_{i}.csv",index=False)
    np.save(f"vset_{i}",tr_set)
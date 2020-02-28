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
    ls = [0,1,2,3,4]
    ls.remove(i)
    validation_index = train_df[train_df['fold']==i].index.values
    training_index = train_df[train_df['fold'].isin(ls)].index.values
    validation_set = x[validation_index]
    training_set = x[training_index]
    val_labels = train_df.loc[validation_index]
    val_labels.to_csv(f"validation_{i}.csv",index=False)
    train_labels = train_df.loc[training_index]
    train_labels.to_csv(f"training_{i}.csv",index=False)
    np.save(f"validation_set_{i}",validation_set)
    np.save(f"training_set_{i}",training_set)
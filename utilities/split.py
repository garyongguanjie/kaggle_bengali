# Adapted from https://www.kaggle.com/yiheng/iterative-stratification
import numpy as np
import pandas as pd

seed = 12
x = np.load("train.npy")

def make_iter_splits(nfold):
    train_df = pd.read_csv('bengaliai-cv19/train.csv')
    train_df = train_df.sample(frac=1)
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
    return train_df


def make_data(csv_name,npy_name,index,df):
    dataset = x[index]
    tr_labels = df.loc[index]
    tr_labels.to_csv(csv_name,index=False)
    np.save(npy_name,dataset)




train_df = make_iter_splits(10)

test_index = train_df[train_df['fold']==3].index.values
val_index = train_df[train_df['fold']==7].index.values
train_index = train_df[train_df['fold'].isin([0,1,2,4,5,6,8,9])].index.values

make_data("shuffle_train_label.csv","shuffle_train",train_index,train_df)
make_data("shuffle_val_label.csv","shuffle_val",val_index,train_df)
make_data("shuffle_test_label.csv","shuffle_test",test_index,train_df)

train_df = make_iter_splits(7)
test_index = train_df[train_df['fold']==2].index.values
val_index = train_df[train_df['fold']==4].index.values
train_index = train_df[train_df['fold'].isin([0,1,3,5,6])].index.values

make_data("shuffle_train_label2.csv","shuffle_train2",train_index,train_df)
make_data("shuffle_val_label2.csv","shuffle_val2",val_index,train_df)
make_data("shuffle_test_label2.csv","shuffle_test2",test_index,train_df)


# for i in range(5):
#     index = train_df[train_df['fold']==i].index.values
#     tr_set = x[index]
#     tr_labels = train_df.loc[index]
#     tr_labels.to_csv(f"set_{i}.csv",index=False)
#     np.save(f"vset_{i}",tr_set)
import pandas as pd
import numpy as np
import glob
import random
import torch
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed: int = 1234):
    """set a fix random seed.
    
    Args:
        seed (int, optional): random seed. Defaults to 9.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class NormalDataset(data.Dataset):

    def __init__(self, data):
        self.x = torch.from_numpy(data).float()[:, :-4].to(device)
        self.y = torch.from_numpy(data).float()[:, -4:].to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def load_data(path, postfix, choose=None):
    files = sorted(glob.glob(path + postfix))
    # print(files)
    if type(choose) is int:
        # random_choose = np.random.randint(0, len(files))
        print(f"building: {files[choose]}")
        df = pd.read_csv(files[choose])
        return df
    elif type(choose) is str:
        file = glob.glob(path + choose + postfix)[0]
        df = pd.read_csv(file)
        return df
    elif type(choose) is list:
        dfs = []
        for file in choose:
            if type(file) is int:
                print(f"building: {files[file]}")
                dfs.append(pd.read_csv(files[file]))
            elif type(file) is str:
                file = glob.glob(path + file + postfix)[0]
                dfs.append(pd.read_csv(file))
        return dfs
    elif choose is None:
        random_choose = np.random.randint(0, len(files))
        # print(f"building: {files[random_choose]}")
        df = pd.read_csv(files[random_choose])
        return df
    else:
        dfs = []
        for file in files:
            dfs.append(pd.read_csv(file))
        return dfs, files
 
def series_to_supervised(data: pd.DataFrame,
                         n_in: int = 1,
                         n_out: int = 1,
                         rate_in: int = 1,
                         rate_out: int = 1,
                         skip: int = 0,
                         sel_in: list = None,
                         sel_out: list = None,
                         dropnan: bool = True):
    """Time series data to supervised data

    Args:
        data (pd.DataFrame): time series data
        n_in (int, optional): input lag. Defaults to 1.
        n_out (int, optional): output lead. Defaults to 1.
        rate_in (int, optional): lag rate. Defaults to 1.
        rate_out (int, optional): lead rate. Defaults to 1.
        skip (int, optional): skip step. Defaults to 0.
        sel_in (list, optional): selected column to input. Defaults to None.
        sel_out (list, optional): selected column to output. Defaults to None.
        dropnan (bool, optional): drop nan or not. Defaults to True.

    Returns:
        df: supervised data
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    orig_cols = df.columns
    cols, names = list(), list()
    # input sequence (t-n, ... t-1) n=n_in
    for i in range(n_in, 0, -rate_in):
        if sel_in is None:
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (orig_cols[j], i)) for j in range(n_vars)]
        else:
            for var in sel_in:
                cols.append(df[var].shift(i))
                names += [('%s(t-%d)' % (var, i))]

    # forecast sequence (t+x, t+x+1, ... t+x+n) x=skip n=n_out
    for i in range(0 + skip, n_out + skip, rate_out):
        if sel_out is None:
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s(t)' % (orig_cols[j])) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (orig_cols[j], i))
                          for j in range(n_vars)]
        else:
            for var in sel_out:
                cols.append(df[var].shift(-i))
                names += [('%s(t+%d)' % (var, i))]
                
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    agg.index = range(len(agg))
    return agg

def construct_dataset(df, dfs, batchsize=32):
    
    train_dfs = []
    test_dfs = []
    
    # divide the dataset (1 year for training, 0.5 year for testing)
    # BDG2 contains 2-year data
    # CBTs contains 1.5-year data
    for i in range(len(dfs)):
        dfs_i = dfs[i]
        n = len(dfs_i)
        train_dfs_i = dfs_i[int(0*n):int(0.5*n)]
        test_dfs_i = dfs_i[int(0.5*n):int(0.75*n)]
        train_dfs.append(train_dfs_i)
        test_dfs.append(test_dfs_i)
    train_dfs = pd.concat(train_dfs)
    test_dfs = pd.concat(test_dfs)

    n = len(df)
    train_df = df[int(0*n):int(0.5*n)]
    test_df = df[int(0.5*n):int(0.75*n)]


    # construct time_series
    train_dfs = series_to_supervised(train_dfs,
                                    n_in=24*1,
                                    n_out=4,
                                    rate_in=1,
                                    rate_out=1,
                                    skip = 0,
                                    sel_in=['value'],
                                    sel_out=['month', 'weekday', 'day', 'hour', 'value'])


    test_dfs = series_to_supervised(test_dfs,
                                n_in=24*1,
                                n_out=4,
                                rate_in=1,
                                rate_out=1,
                                skip = 0,
                                sel_in=['value'],
                                sel_out=['month', 'weekday', 'day', 'hour', 'value'])
    
    train_ds = series_to_supervised(train_df,
                                    n_in=24*1,
                                    n_out=4,
                                    rate_in=1,
                                    rate_out=1,
                                    skip = 0,
                                    sel_in=['value'],
                                    sel_out=['month', 'weekday', 'day', 'hour', 'value'])

    test_ds = series_to_supervised(test_df,
                                n_in=24*1,
                                n_out=4,
                                rate_in=1,
                                rate_out=1,
                                skip = 0,
                                sel_in=['value'],
                                sel_out=['month', 'weekday', 'day', 'hour', 'value'])
    
    train_ds = train_ds.values
    test_ds = test_ds.values
    train_dfs = train_dfs.values
    test_dfs = test_dfs.values

    cols_to_move = [28,33,38,43]
    remaining_cols = [i for i in range(train_ds.shape[1]) if i not in cols_to_move]
    new_order = remaining_cols + cols_to_move
    train_ds = train_ds[:, new_order]
    train_dfs = train_dfs[:, new_order]
    test_ds = test_ds[:, new_order]
    test_dfs = test_dfs[:, new_order]

    scalerx = StandardScaler()
    scalery = StandardScaler()
    scalerx = scalerx.fit(train_dfs[:, :-4])
    scalery = scalery.fit(train_dfs[:,-4:])

    train_ds[:, :-4] = scalerx.transform(train_ds[:, :-4])
    test_ds[:, :-4] = scalerx.transform(test_ds[:, :-4])

    train_ds[:,-4:] = scalery.transform(train_ds[:,-4:])
    test_ds[:,-4:] = scalery.transform(test_ds[:,-4:])

    train_dfs[:, :-4] = scalerx.transform(train_dfs[:, :-4])
    test_dfs[:, :-4] = scalerx.transform(test_dfs[:, :-4])

    train_dfs[:,-4:] = scalery.transform(train_dfs[:,-4:])
    test_dfs[:,-4:] = scalery.transform(test_dfs[:,-4:])

    train_ds = NormalDataset(train_ds)
    test_ds = NormalDataset(test_ds)
    train_dfs = NormalDataset(train_dfs)
    test_dfs = NormalDataset(test_dfs)
    
    input_dim = train_ds.x[0].shape[-1]

    train_dataloader = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    test_dataloader = data.DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    global_train_dataloader = data.DataLoader(train_dfs, batch_size=batchsize, shuffle=True)
    global_test_dataloader = data.DataLoader(test_dfs, batch_size=len(test_dfs), shuffle=False)

    return train_dataloader, input_dim, test_dataloader, scalery, input_dim, global_train_dataloader, input_dim, global_test_dataloader

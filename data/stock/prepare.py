import os
import pickle
import numpy as np
import pandas as pd

# # get stock datasets
# train_path = './datasets/train/'
# val_path = './datasets/val/'

# train_files = os.listdir(train_path)
# val_files = os.listdir(val_path)

# #根据时间将多个股票的数据合并
# train_data = []
# val_data = []
# test = -1
# for idx, file in enumerate(train_files):
#     df = pd.read_csv(f'{train_path}/{file}')
#     #只保留data 和 profile
#     df = df[['date', 'profile']]
#     #将profile 改名为文件名
#     df = df.rename(columns={'profile': file[:-4]})
#     train_data.append(df)
#     if idx == test:
#         break
# #将相同时间的数据设为同一行
# train_data = pd.concat(train_data, axis=1)
# #fill NaN with 0
# train_data = train_data.fillna(0)
# #numpy
# train_data = train_data.to_numpy()
# train_data = train_data[:, 1::2]
# train_data = train_data.astype(np.float32)

# for idx, file in enumerate(val_files):
#     df = pd.read_csv(f'{val_path}/{file}')
#     df = df[['date', 'profile']]
#     df = df.rename(columns={'profile': file[:-4]})
#     val_data.append(df)
#     if idx == test:
#         break
# val_data = pd.concat(val_data, axis=1)
# val_data = val_data.fillna(0)
# val_data = val_data.to_numpy()
# val_data = val_data[:, 1::2]
# val_data = val_data.astype(np.float32)

# #save the data
# np.save(os.path.join(os.path.dirname(__file__), 'train.npy'), train_data)
# np.save(os.path.join(os.path.dirname(__file__), 'val.npy'), val_data)

#load
train_data = np.load(os.path.join(os.path.dirname(__file__), 'train.npy'))
val_data = np.load(os.path.join(os.path.dirname(__file__), 'val.npy'))


# get all the unique characters that occur in this text
vocab_size = 402

THRES = np.arange(-10000, 10050, 50)
def encode(s: np.ndarray):
    a = s * 10000
    #a <= THRES[idx]
    l = np.zeros_like(a, dtype=np.uint16)
    for idx in reversed(range(len(THRES) - 1)):
        l[np.where(a <= THRES[idx])] = idx
    l[np.where(a > THRES[-1])] = vocab_size - 1
    return l


def decode(l):
    s = np.zeros_like(l, dtype=np.float32)
    s[np.where(l == vocab_size - 1)] = THRES[-1] / 10000
    for idx in range(len(THRES) - 1):
        s[np.where(l == idx)] = THRES[idx] / 10000
    return s



# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
# save as npy
np.save(os.path.join(os.path.dirname(__file__), 'train_ids.npy'), train_ids)
np.save(os.path.join(os.path.dirname(__file__), 'val_ids.npy'), val_ids)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
itos = {i: decode(np.ones(1) * i).item() for i in range(vocab_size)}
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    # 'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens

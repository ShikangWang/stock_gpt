import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
num_samples = 1 # number of samples to draw
max_new_tokens = 1 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

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

block_size = 256
# load
val_data = np.load(os.path.join('data/stock', 'val.npy'))
val_ids = np.load(os.path.join('data/stock', 'val_ids.npy')).astype(np.int64)

stock_nums = val_data.shape[1]
num_days = val_data.shape[0]

with torch.no_grad():
    with ctx:
        profiles = []
        for day in tqdm(range(num_days - block_size)):
            preds = []
            # for stock in range(stock_nums):
            #     x = torch.tensor(val_ids[day:day + block_size, stock], dtype=torch.long, device=device)[None, ...]
            #     y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            #     preds.append(y[0, -1].item())
            # stock 并行
            x = torch.tensor(val_ids[day:day + block_size], dtype=torch.long, device=device)
            x = x.permute(1, 0)
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            preds = y[:, -1].cpu().numpy()


            #get the idx of the 10% highest stock
            preds = np.array(preds)
            invest_num = min(int(stock_nums * 0.05), sum(preds > 200))
            if sum(preds > 200) < int(stock_nums * 0.05):
                print('Warning: not enough stocks to invest')
            idx = np.argsort(preds)[-int(invest_num):]
            #get the true profile of these stocks
            gt = val_data[day + block_size]
            profile = gt[idx]
            profile = np.mean(profile)
            profiles.append(profile)
            #empty the cache
            torch.cuda.empty_cache()
            print(profile * 100)
        profiles = np.array(profiles)
        #all the profiles
        print(np.mean(profiles))
        print(np.std(profiles))
        print(np.min(profiles))
        print(np.max(profiles))
        #总利润
        print(np.sum(profiles))
        
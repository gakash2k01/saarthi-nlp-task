import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import train
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import random,yaml
import functools
import pathlib,os,sys
import warnings
warnings.filterwarnings("ignore")

# +
# To make all paths relative
base_path = pathlib.Path().absolute()

# Importing configurations
yml_path = f"{base_path}/config/config.yml"
with open(yml_path, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
model_name = cfg["params"]["model_name"]

# Loading model and pretrained weights
print("Loading model and weights...")
model = T5ForConditionalGeneration.from_pretrained(model_name)
checkpoint = torch.load(f"{base_path}/weights/best_model.pth")
model.load_state_dict(checkpoint)
model.eval()
model.to('cpu')
tokenizer = T5Tokenizer.from_pretrained(model_name)


# -

def tokenize(text):
    '''
    Function: converts words to tokens.
    Input: Word
    Output: tokens, attention-mask
    '''
    res = tokenizer.encode_plus(text, padding="max_length")
    return torch.tensor(res.input_ids), torch.tensor(res.attention_mask)


def predict(sentence, model):
    '''
    Function: Prediction
    Input: sentence, model
    Output: NIL
    '''
    inp_ids, inp_mask = tokenize(sentence)
    inp_ids = inp_ids.unsqueeze(0)
    inp_ids = inp_ids.to('cpu')
    with torch.no_grad():
        output = model.generate(input_ids = inp_ids)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


# Type the sentence to be processed.
sentence = "Dog is barking."
predict(sentence, model)

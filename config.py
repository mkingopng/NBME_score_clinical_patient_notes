"""

"""
# libraries
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
import wandb

from IPython.core.display_functions import display
import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoModel, DistilBertTokenizerFast

# constants
MODEL = AutoModel.from_pretrained('distilbert-base-uncased')
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
CONFIG = AutoConfig.from_pretrained('distilbert-base-uncased')
OUTPUT_DIR = 'output_dir'
SEED = 42
DATA_DIR = 'data'
TRAIN = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
FEATURES = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'))
TEST = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
PATIENT_NOTES = pd.read_csv(os.path.join(DATA_DIR, 'patient_notes.csv'))


# configuration
class CONFIGURATION:
    wandb = False
    competition = 'NBME'
    _wandb_kernel = 'mkingo'
    debug = False
    apex = True
    print_freq = 100
    num_workers = 4
    model = MODEL
    tokenizer = TOKENIZER
    config = CONFIG
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 5
    encoder_lr = 2e-5  # mk: the use of encoder & decoder implies that this isn't straight NER
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 4
    fc_dropout = 0.2
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 5
    trn_fold = [0]
    train = True


if CONFIGURATION.debug:
    CONFIGURATION.epochs = 2
    CONFIGURATION.trn_fold = [0]


# options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
TOKENIZERS_PARALLELISM = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

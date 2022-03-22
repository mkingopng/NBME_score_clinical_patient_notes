"""
libraries
"""
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
import wandb
from wandb_creds import *
from transformers import AutoModel, DistilBertTokenizerFast

"""
constants & options
"""
SEED = 42
OUTPUT_DIR = 'EXPERIMENT_1_'  # increment for each iteration
MODEL = AutoModel.from_pretrained('distilbert-base-uncased')
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
CONFIGURATION
"""


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
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 5
    encoder_lr = 2e-5
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


"""
wandb
"""

if CONFIGURATION.wandb:
    wandb.login(key=API_KEY)

    def class2dict(f):
        """

        :param f:
        :return:
        """
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    run = wandb.init(
        project=CONFIGURATION.competition,
        name=CONFIGURATION.model,
        config=class2dict(CONFIGURATION),
        group=CONFIGURATION.model,
        job_type="train",
    )

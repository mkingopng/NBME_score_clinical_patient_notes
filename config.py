"""
Library imports
"""
from wandb_creds import *
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
from DeBERTa import deberta

"""
environmental variables
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 42
DATA_DIR = 'kaggle/input/nbme-score-clinical-patient-notes'
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
OUTPUT_DIR = 'deberta_large_'   # iterate for each experiment

"""
CONFIGURATION
"""


class CONFIGURATION:
    apex = True
    batch_scheduler = True
    batch_size = 4
    betas = (0.9, 0.999)
    competition = 'NBME'
    decoder_lr = 3e-5
    debug = False
    epochs = 10
    encoder_lr = 2e-5
    gradient_accumulation_steps = 1
    min_lr = 1e-6
    eps = 1e-6
    fc_dropout = 0.2
    max_grad_norm = 1000
    max_len = 512
    model = 'microsoft/deberta-large'  # deberta base, deberta large, roberta, t5, bert,
    n_fold = 5
    num_cycles = 0.5
    num_warmup_steps = 0
    num_workers = 4
    print_freq = 100
    weight_decay = 0.01
    scheduler = 'cosine'
    seed = 42
    trn_fold = [0, 1, 2, 3, 4]
    train = True
    tokenizer = AutoTokenizer.from_pretrained(model)
    wandb = True
    _wandb_kernel = ENTITY


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
    # sweep_id = wandb.sweep(sweep_config, project="nbme_sweeps_testing")


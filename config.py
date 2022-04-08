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
# import models_and_tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5TokenizerFast, DebertaTokenizerFast
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM
from DeBERTa import deberta

"""
environmental variables
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 42  # random.seed(42)  # fix_me
DATA_DIR = 'kaggle/input/nbme-score-clinical-patient-notes'
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
OUTPUT_DIR = 'checkpoints_'  # heading: iterate for each experiment

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
    epochs = 5
    encoder_lr = 2e-5
    gradient_accumulation_steps = 1
    min_lr = 1e-6
    eps = 1e-6
    fc_dropout = 0.2
    max_grad_norm = 1000
    max_len = 512
    model = 'bert-base-uncased'
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


class Checkpoints:
    # deberta_base_checkpoints = 'kaggle/inputs/checkpoints/deberta_base'
    # deberta_large_checkpoints = 'kaggle/inputs/checkpoints/deberta_large'
    # deberta_v2_checkpoints = 'kaggle/inputs/checkpoints/deberta_v2'
    deberta_v3_checkpoints = 'kaggle/inputs/checkpoints/deberta_v3'
    # roberta_base_checkpoints = 'kaggle/inputs/checkpoints/roberta_base'
    # roberta_large_checkpoints = 'kaggle/inputs/checkpoints/roberta_large'
    # t5_checkpoints = 'kaggle/inputs/checkpoints/t5'
    # bert_base_uncased_checkpoints = 'kaggle/inputs/checkpoints/bert_base_uncased'
    # distilgpt2_checkpoints = "kaggle/inputs/checkpoints/distilgpt2"


class Model:
    # deberta_base_model = AutoModel.from_pretrained('kaggle/input/models/deberta_base_model')
    # deberta_large_model = AutoModel.from_pretrained('kaggle/input/models/deberta_large_model')
    # deberta_v2_xlarge_model = AutoModel.from_pretrained('kaggle/input/models/deberta_v2_xlarge_model')
    # deberta_v3_large_model = AutoModel.from_pretrained('kaggle/input/models/deberta_v3_large_model')
    # roberta_base_model = AutoModel.from_pretrained('kaggle/input/models/roberta_base_model')
    # roberta_large_model = AutoModel.from_pretrained('kaggle/input/models/roberta_large_model')
    # t5_base_model = AutoModel.from_pretrained("kaggle/input/models/t5_base_tokenizer")
    # t5_large_model = AutoModel.from_pretrained("kaggle/input/models/t5_large_tokenizer")
    bert_base_uncased_model = AutoModel.from_pretrained("kaggle/input/models/bert_base_uncased_model")
    # distilpgt2_model = AutoModel.from_pretrained("kaggle/input/models/distilgpt2_model")


class Tokenizer:
    # deberta_base_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/tokenizers/bert_base_uncased_tokenizer"),
    # deberta_large_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/tokenizers/deberta_large_tokenizer"),
    # deberta_v2_tokenizer = T5TokenizerFast("T5"),
    # deberta_v3_large_tokenizer = DebertaTokenizerFast(
    #                                         vocab_file=,
    #                                          tokenizer_file=,
    #                                          eos_token=,
    #                                          unk_token=,
    #                                          pad_token=,
    #                                          extra_ids=,
    #                                          additional_special_tokens=,
    #                                          ),
    # roberta_base_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/tokenizers/roberta_base_tokenizer"),
    # roberta_large_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/tokenizers/roberta_large_tokenizer")
    # t5_base_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/tokenizers/t5_base_tokenizer"),
    # t5_large_tokenizer = T5TokenizerFast(vocab_file=,
    #                                      do_lower_case=,
    #                                      unk_token=,
    #                                      sep_token=,
    #                                      pad_token=,
    #                                      cls_token=,
    #                                      mask_token=,
    #                                      ),
    bert_base_uncased_tokenizer = AutoTokenizer.from_pretrained(
        "kaggle/input/tokenizers/bert_base_uncased_tokenizer"),
    # distilgpt2_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/tokenizers/distilgpt2_tokenizer")


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

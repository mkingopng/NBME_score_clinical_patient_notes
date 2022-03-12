"""

"""
# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3

import os
from datetime import datetime
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from itertools import chain
from functools import partial
from ast import literal_eval
import shutil
import wandb


import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.offline as pyo

from datasets import load_dataset
from datasets import Dataset

import transformers
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import HfArgumentParser
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed
from transformers import logging
from transformers import ModelArguments
from transformers import DataTrainingArguments

from transformers.modeling_outputs import TokenClassifierOutput


transformers_path = Path("t")

input_dir = Path("db_v2_3_fast_tokenizer")

convert_file = input_dir / "convert_slow_tokenizer.py"

deberta_v2 = "deberta_v2"

DEBUG = False

all_models = ['microsoft/deberta-v3-base'] * 5

model_args = ModelArguments(model_name_or_path=all_models[0])

data_args = DataTrainingArguments(
    k_folds=5,
    max_seq_length=512,
    pad_to_max_length=False,
    preprocessing_num_workers=4,
)

training_args = TrainingArguments(
    output_dir="model",
    do_train=True,
    do_eval=True,
    do_predict=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    logging_steps=75,
    evaluation_strategy="epoch",
    save_strategy="no",
    seed=18,
    fp16=False,
    report_to="wandb",
    group_by_length=True,
)

set_seed(training_args.seed)

data_dir = "data"
feats_df = pd.read_csv(os.path.join(data_dir, "features.csv"))
notes_df = pd.read_csv(os.path.join(data_dir, "patient_notes.csv"))
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

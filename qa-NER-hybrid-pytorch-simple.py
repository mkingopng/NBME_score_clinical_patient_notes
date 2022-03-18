"""
This script is interesting but has abug that needs fixing.
"""

import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import StratifiedKFold
import os
import warnings
from datetime import datetime
from collections import Counter
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from itertools import chain
from functools import partial
from ast import literal_eval
import torch.nn as f
import torch
from sklearn.metrics import precision_recall_fscore_support
# import plotly.express as px
# import plotly.offline as pyo
# pyo.init_notebook_mode()
import pandas as pd
import numpy as np
# from datasets import load_dataset, Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    logging,
)
from transformers.modeling_outputs import TokenClassifierOutput

# environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
logging.set_verbosity(logging.WARNING)
TOKENIZERS_PARALLELISM = True

# data
df = pd.read_csv("data/features.csv")
print(df.info())
print(df.head())

notes = pd.read_csv('data/patient_notes.csv')
print(notes.info())
print(notes.head(3))

train = pd.read_csv("data/train.csv")
print(train.info())
print(train.head(3))

sample = pd.read_csv("data/sample_submission.csv")
print(sample.info())
print(sample.head(3))

# eda
print(df.case_num.value_counts())

for y in range(10):
    for x in df[df.case_num == y][['case_num', 'feature_text']].values:
        print(x[0], x[1])
    print("*" * 50)

# patient notes
print(notes.columns)

print(len(notes.pn_num.value_counts()) == notes.shape[0])

print(len(notes.case_num.value_counts() == notes.shape[0]))

print(len(notes.case_num.value_counts()))

plt.bar(notes.groupby('case_num').count().index, notes.groupby('case_num').count()['pn_num'])

for x in notes.groupby('case_num').get_group(0.0)[["case_num", 'pn_history']].values[:3]:
    print(x[0], ":-", x[1])
    print("*" * 50)
    print("\n")

print(notes.shape)

print(df.columns)

print(notes.columns)

new = pd.merge(df, notes, on='case_num', how="inner")
print(new.shape)

# train
print(train.columns)

print(train.shape)
print(train.head(5))

cn = train.groupby('case_num').get_group(1.0)
print(cn.shape)
print(cn.head())

# missing stuff
missing_annotations = train["annotation"] == "[]"
missing_locations = train["location"] == "[]"
both_missing = (train["annotation"] == train["location"]) & missing_annotations

sum(missing_annotations), sum(missing_locations), sum(both_missing)

skf = StratifiedKFold(n_splits=5, random_state=18, shuffle=True)

splits = list(skf.split(X=notes, y=notes['case_num']))

notes["fold"] = -1

for fold, (_, val_idx) in enumerate(skf.split(notes, y=notes["case_num"])):
    notes.loc[val_idx, "fold"] = fold

counts = notes.groupby(["fold", "pn_num"], as_index=False).count()

# If the number of rows is the same as the number of  unique pn_num, then each pn_num is only in one fold. Also if all
# the counts=1
print(counts.shape, counts.pn_num.nunique(), counts.case_num.unique())
print(counts)

t = train.merge(notes, how="left").merge(df, how='left')

merged = train.merge(notes, how="left")
merged = merged.merge(df, how="left")

merged.head(10)

merged.loc[338, "anno_list"] = '["father heart attack"]'
merged.loc[338, "loc_list"] = '["764 783"]'

merged.loc[621, "anno_list"] = '["for the last 2-3 months", "over the last 2 months"]'
merged.loc[621, "loc_list"] = '["77 100", "398 420"]'

merged.loc[655, "anno_list"] = '["no heat intolerance", "no cold intolerance"]'
merged.loc[655, "loc_list"] = '["285 292;301 312", "285 287;296 312"]'

merged.loc[1262, "anno_list"] = '["mother thyroid problem"]'
merged.loc[1262, "loc_list"] = '["551 557;565 580"]'

merged.loc[1265, "anno_list"] = '[\'felt like he was going to "pass out"\']'
merged.loc[1265, "loc_list"] = '["131 135;181 212"]'

merged.loc[1396, "anno_list"] = '["stool , with no blood"]'
merged.loc[1396, "loc_list"] = '["259 280"]'

merged.loc[1591, "anno_list"] = '["diarrhoe non blooody"]'
merged.loc[1591, "loc_list"] = '["176 184;201 212"]'

merged.loc[1615, "anno_list"] = '["diarrhea for last 2-3 days"]'
merged.loc[1615, "loc_list"] = '["249 257;271 288"]'

merged.loc[1664, "anno_list"] = '["no vaginal discharge"]'
merged.loc[1664, "loc_list"] = '["822 824;907 924"]'

merged.loc[1714, "anno_list"] = '["started about 8-10 hours ago"]'
merged.loc[1714, "loc_list"] = '["101 129"]'

merged.loc[1929, "anno_list"] = '["no blood in the stool"]'
merged.loc[1929, "loc_list"] = '["531 539;549 561"]'

merged.loc[2134, "anno_list"] = '["last sexually active 9 months ago"]'
merged.loc[2134, "loc_list"] = '["540 560;581 593"]'

merged.loc[2191, "anno_list"] = '["right lower quadrant pain"]'
merged.loc[2191, "loc_list"] = '["32 57"]'

merged.loc[2553, "anno_list"] = '["diarrhoea no blood"]'
merged.loc[2553, "loc_list"] = '["308 317;376 384"]'

merged.loc[3124, "anno_list"] = '["sweating"]'
merged.loc[3124, "loc_list"] = '["549 557"]'

merged.loc[
    3858, "anno_list"] = '["previously as regular", "previously eveyr 28-29 days", "previously lasting 5 days", "previously regular flow"]'
merged.loc[3858, "loc_list"] = '["102 123", "102 112;125 141", "102 112;143 157", "102 112;159 171"]'

merged.loc[4373, "anno_list"] = '["for 2 months"]'
merged.loc[4373, "loc_list"] = '["33 45"]'

merged.loc[4763, "anno_list"] = '["35 year old"]'
merged.loc[4763, "loc_list"] = '["5 16"]'

merged.loc[4782, "anno_list"] = '["darker brown stools"]'
merged.loc[4782, "loc_list"] = '["175 194"]'

merged.loc[4908, "anno_list"] = '["uncle with peptic ulcer"]'
merged.loc[4908, "loc_list"] = '["700 723"]'

merged.loc[6016, "anno_list"] = '["difficulty falling asleep"]'
merged.loc[6016, "loc_list"] = '["225 250"]'

merged.loc[6192, "anno_list"] = '["helps to take care of aging mother and in-laws"]'
merged.loc[6192, "loc_list"] = '["197 218;236 260"]'

merged.loc[
    6380, "anno_list"] = '["No hair changes", "No skin changes", "No GI changes", "No palpitations", "No excessive sweating"]'
merged.loc[
    6380, "loc_list"] = '["480 482;507 519", "480 482;499 503;512 519", "480 482;521 531", "480 482;533 545", "480 482;564 582"]'

merged.loc[
    6562, "anno_list"] = '["stressed due to taking care of her mother", "stressed due to taking care of husbands parents"]'
merged.loc[6562, "loc_list"] = '["290 320;327 337", "290 320;342 358"]'

merged.loc[6862, "anno_list"] = '["stressor taking care of many sick family members"]'
merged.loc[6862, "loc_list"] = '["288 296;324 363"]'

merged.loc[7022, "anno_list"] = '["heart started racing and felt numbness for the 1st time in her finger tips"]'
merged.loc[7022, "loc_list"] = '["108 182"]'

merged.loc[7422, "anno_list"] = '["first started 5 yrs"]'
merged.loc[7422, "loc_list"] = '["102 121"]'

merged.loc[8876, "anno_list"] = '["No shortness of breath"]'
merged.loc[8876, "loc_list"] = '["481 483;533 552"]'

merged.loc[9027, "anno_list"] = '["recent URI", "nasal stuffines, rhinorrhea, for 3-4 days"]'
merged.loc[9027, "loc_list"] = '["92 102", "123 164"]'

merged.loc[
    9938, "anno_list"] = '["irregularity with her cycles", "heavier bleeding", "changes her pad every couple hours"]'
merged.loc[9938, "loc_list"] = '["89 117", "122 138", "368 402"]'

merged.loc[9973, "anno_list"] = '["gaining 10-15 lbs"]'
merged.loc[9973, "loc_list"] = '["344 361"]'

merged.loc[10513, "anno_list"] = '["weight gain", "gain of 10-16lbs"]'
merged.loc[10513, "loc_list"] = '["600 611", "607 623"]'

merged.loc[11551, "anno_list"] = '["seeing her son knows are not real"]'
merged.loc[11551, "loc_list"] = '["386 400;443 461"]'

merged.loc[11677, "anno_list"] = '["saw him once in the kitchen after he died"]'
merged.loc[11677, "loc_list"] = '["160 201"]'

merged.loc[12124, "anno_list"] = '["tried Ambien but it didnt work"]'
merged.loc[12124, "loc_list"] = '["325 337;349 366"]'

merged.loc[
    12279, "anno_list"] = '["heard what she described as a party later than evening these things did not actually happen"]'
merged.loc[12279, "loc_list"] = '["405 459;488 524"]'

merged.loc[
    12289, "anno_list"] = '["experienced seeing her son at the kitchen table these things did not actually happen"]'
merged.loc[12289, "loc_list"] = '["353 400;488 524"]'

merged.loc[13238, "anno_list"] = '["SCRACHY THROAT", "RUNNY NOSE"]'
merged.loc[13238, "loc_list"] = '["293 307", "321 331"]'

merged.loc[
    13297, "anno_list"] = '["without improvement when taking tylenol", "without improvement when taking ibuprofen"]'
merged.loc[13297, "loc_list"] = '["182 221", "182 213;225 234"]'

merged.loc[13299, "anno_list"] = '["yesterday", "yesterday"]'
merged.loc[13299, "loc_list"] = '["79 88", "409 418"]'

merged.loc[13845, "anno_list"] = '["headache global", "headache throughout her head"]'
merged.loc[13845, "loc_list"] = '["86 94;230 236", "86 94;237 256"]'

merged.loc[14083, "anno_list"] = '["headache generalized in her head"]'
merged.loc[14083, "loc_list"] = '["56 64;156 179"]'

merged["anno_list"] = [literal_eval(x) if isinstance(x, str) else x for x in merged["annotation"]]
merged["loc_list"] = [literal_eval(x) if isinstance(x, str) else x for x in merged["location"]]

merged = merged[merged["anno_list"].map(len) != 0].copy().reset_index(drop=True)

print(merged.head())


def process_feature_text(text):
    return text.replace("-", " ")


merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]

print(merged.shape)

merged["anno_list"] = [literal_eval(x) if isinstance(x, str) else x for x in merged["annotation"]]
merged["loc_list"] = [literal_eval(x) if isinstance(x, str) else x for x in merged["location"]]

merged = merged[merged["anno_list"].map(len) != 0].copy().reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("roberta-large")


def fn(x):
    return len(x.split())


def loc_list_to_ints(loc_list):
    to_return = []

    for loc_str in loc_list:
        loc_strs = loc_str.split(";")

        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))

    return to_return


def process_feature_text(text):
    return text.replace("-", " ")


def tokenize_and_add_labels(example, tokenizer=tokenizer):
    tokenized_inputs = tokenizer(
        example["feature_text"],
        example["text"],
        truncation="only_second",
        max_length=416,
        padding="max_length",
        return_offsets_mapping=True,

    )

    # labels should be float
    labels = [0.0] * len(tokenized_inputs["input_ids"])
    tokenized_inputs["locations"] = loc_list_to_ints(example["loc_list"])
    tokenized_inputs["sequence_ids"] = [0 if i == 0 or i == None else 1 for i in tokenized_inputs.sequence_ids()]

    for idx, (seq_id, offsets) in enumerate(zip(tokenized_inputs["sequence_ids"], tokenized_inputs["offset_mapping"])):
        if seq_id is None or seq_id == 0:
            labels[idx] = -100.0  # don't calculate loss on question part or special tokens
            continue

        exit = False
        token_start, token_end = offsets
        for feature_start, feature_end in tokenized_inputs["locations"]:
            if exit:
                break
            if token_start <= feature_start < token_end or token_start < feature_end <= token_end or feature_start <= token_start < feature_end:
                labels[idx] = 1.0  # labels should be float
                exit = True

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


merged.rename(columns={'pn_history': 'text'}, inplace=True)

first = merged[merged['fold'] == 0].iloc[37]
example = {
    "feature_text": first.feature_text,
    "text": first.text,
    "loc_list": first.loc_list,
    "annotations": first.anno_list,
}
print(example, "\n\n")
tokenized = partial(tokenize_and_add_labels, tokenizer=tokenizer)(example)

tokens = tokenizer.tokenize(example["feature_text"], example["text"], add_special_tokens=True)

print("Locations")
print(example["loc_list"], "\n")

print("Annotations")
print(example["annotations"], "\n")

print("Token | Label | Token Offsets")
zipped = list(zip(tokens, tokenized["labels"], tokenized["offset_mapping"]))
[x for x in zipped if x[1] > 0]

print(tokenized)

print(merged.columns)


def tokenize(idx, tensor=True):
    l1, l2, l3, l4, l5, l6 = [], [], [], [], [], []
    for x in idx:
        first = merged[['feature_text', "text", 'loc_list', 'anno_list', 'pn_num']].iloc[x].values
        example = {
            "feature_text": first[0],
            "text": first[1],
            "loc_list": first[2],
            "annotations": first[3],
            "pn_num": first[4]

        }

        dict1 = tokenize_and_add_labels(example)
        # l4.append(example['text'])
        l1.append(dict1['input_ids'])
        l2.append(dict1['attention_mask'])
        l3.append(dict1['offset_mapping'])
        l4.append(example['pn_num'])
        # l4.append(dict1['locations']) 'location':l4
        l5.append(dict1['sequence_ids'])
        l6.append(dict1['labels'])
    encoded = {'input_ids': l1, 'attention_mask': l2, 'offset_mapping': l3, 'sequence_ids': l5, 'labels': l6,
               'pn_num': l4}

    if tensor:
        encoded = {key: torch.as_tensor(val) for key, val in encoded.items()}
    return encoded


fold_0 = merged[merged['fold'] == 0].index
fold_0 = tokenize(fold_0, tensor=True)
print(len(fold_0['input_ids']))


class scoreDataset(Dataset):
    def __init__(self, tokenized_ds):
        self.data = tokenized_ds

    def __getitem__(self, index):
        item = {k: self.data[k][index] for k in self.data.keys()}
        return item

    def __len__(self):
        return len(self.data['input_ids'])


ds = scoreDataset(fold_0)
ds = DataLoader(ds, batch_size=3,
                shuffle=True, num_workers=0, pin_memory=True)

val = scoreDataset(fold_0)
val = DataLoader(val, batch_size=3,
                 shuffle=False, num_workers=0, pin_memory=True)

for batch in val:
    print(batch['labels'])
    break

# model training
from transformers import DistilBertModel, RobertaModel


class NeuralNetwork(f.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.dense = f.Linear(1024, 1)
        self.backbone = RobertaModel.from_pretrained(
            "roberta-large")  # DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = f.Dropout(.2)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.backbone(input_ids, attention_mask)
        out = self.dropout(out[0])

        out = self.dense(out)
        return out


net = NeuralNetwork()


def sigmoid(z):
    return f.Sigmoid()(z)


def compute_metrics(eval_prediction):
    """
    This only gets the scores at the token level. The actual leaderboard score is based at the character level.
    The CV score at the character level is handled in the evaluate function of the trainer.
    """
    predictions, y_true = eval_prediction
    predictions = sigmoid(predictions)
    # y_true = y_true.astype(int)

    y_pred = [
        [int(p > 0.5) for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, y_true)
    ]

    # Remove ignored index (special tokens)
    y_true = [
        [l for l in label if l != -100]
        for label in y_true
    ]

    results = precision_recall_fscore_support(list(chain(*y_true)), list(chain(*y_pred)), average="binary")
    return {
        "token_precision": results[0],
        "token_recall": results[1],
        "token_f1": results[2]
    }


def get_location_predictions(dataset, preds):
    """
    It's easier to run CV if we don't convert predictions into the format expected at test time.
    """
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, dataset["offset_mapping"], dataset["sequence_ids"]):
        pred = sigmoid(pred)
        start_idx = None
        current_preds = []
        for p, o, s_id in zip(pred, offsets, seq_ids):
            if s_id is None or s_id == 0:
                continue

            if p > 0.5:
                if start_idx is None:
                    start_idx = o[0]
                end_idx = o[1]
            elif start_idx is not None:
                current_preds.append((start_idx, end_idx))
                start_idx = None

        if start_idx is not None:
            current_preds.append((start_idx, end_idx))

        all_predictions.extend(current_preds)

    return all_predictions


def calculate_char_CV(dataset):
    """
    Some tokenizers include the leading space as the start of the offset_mapping, so there is code to ignore that space.
    """
    all_labels = []
    all_preds = []

    for batch in tqdm(dataset):
        with torch.no_grad():
            logits = net(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        l1 = []
        for no in batch['pn_num']:
            l1.append(merged[merged['pn_num'] == no.numpy()].text.iloc[1])

        #         for x in batch['input_ids']:
        #                 text=tokenizer.decode(batch['input_ids'][0]).split()
        #                 start=text.index('[SEP]')
        #                 end=text[start+1:].index('[SEP]')
        #                 text=" ".join(text[start+1:start+end+1])
        #                 l1.append(text)
        batch['text'] = np.array(l1)
        predictions = get_location_predictions(batch, logits)
        #         print("predictions",predictions)
        #         print("\n")
        for preds, offsets, seq_ids, labels, text in zip(
                predictions,
                batch["offset_mapping"],
                batch["sequence_ids"],
                batch["labels"],
                batch["text"]
        ):
            try:
                #                     print("length",len(text))
                #                     print(text)
                #                     print("\n")

                num_chars = max(list(chain(*offsets)))
                char_labels = np.zeros((num_chars))
                count = 0
                for o, s_id, label in zip(offsets, seq_ids, labels):

                    if s_id is None or s_id == 0:  # ignore question part of input
                        continue
                    if int(label) == 1:

                        char_labels[o[0]:o[1]] = 1
                        if text[o[0]].isspace() and o[0] > 0 and char_labels[o[0] - 1] != 1:
                            char_labels[o[0]] = 0
                    count += 1
                char_preds = np.zeros((num_chars))
                #                     print(preds)
                #                     print(preds[0])
                char_preds[preds[0]:preds[1]] = 1
                if text[preds[0]].isspace():
                    char_preds[preds[0]] = 0

                all_labels.extend(char_labels)
                all_preds.extend(char_preds)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(len(text))

    #                     print(precision_recall_fscore_support(all_labels, all_preds, average="binary"))

    results = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2]
    }


net.to(device)


# net.load_state_dict(torch.load("pytorch_model_e4.bin"))

def train(model, optimizer, dl_train, epoch):
    time_start = time.time()

    # Set learning rate to the one in config for this epoch
    for g in optimizer.param_groups:
        g['lr'] = config['learning_rates'][epoch]
    lr = optimizer.param_groups[0]['lr']

    epoch_prefix = f"[Epoch {epoch + 1:2d} / {config['epochs']:2d}]"
    print(f"{epoch_prefix} Starting epoch {epoch + 1:2d} with LR = {lr}")

    # Put model in training mode
    model.train()

    # Accumulator variables
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_f1, tr_p, tr_r = 0, 0, 0
    loop = tqdm(enumerate(dl_train), leave=False, total=2022 // 3)
    for idx, batch in loop:
        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.float)

        tr_logits = model(input_ids=ids, attention_mask=mask, )
        loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fct(tr_logits.view(-1, 1), labels.view(-1, 1))

        # this ignores the part of the sequence that got -100 as labels
        loss = torch.masked_select(loss, labels.view(-1, 1) > -1).mean()
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        loss_step = tr_loss / nb_tr_steps

        #         if idx % 200 == 0:

        #             print(f"{epoch_prefix}     Steps: {idx:4d} --> Loss: {loss_step:.4f}")

        # compute training accuracy

        tmp_tr_accuracy = compute_metrics((tr_logits.to('cpu'), batch['labels'].to('cpu')))
        tr_f1 += tmp_tr_accuracy['token_f1']
        tr_p += tmp_tr_accuracy['token_precision']
        tr_r += tmp_tr_accuracy['token_recall']
        loop.set_description(f"Epoch[{epoch}/{config['epochs']:2d}]")
        loop.set_postfix(loss=loss_step)
        # wandb.log({'Train Loss (Step)': loss_step, 'Train Accuracy (Step)' : tr_accuracy / nb_tr_steps})

        #  gradient clipping
        #         torch.nn.utils.clip_grad_norm_(
        #            parameters=model.parameters(), max_norm=config['max_grad_norm']
        #        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_f1 = tr_f1 / nb_tr_steps
    tr_p = tr_f1 / nb_tr_steps
    tr_r = tr_f1 / nb_tr_steps

    torch.save(model.state_dict(), f'pytorch_model_e{epoch}.bin')
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - time_start

    print(epoch_prefix)
    print(f"{epoch_prefix} Training loss    : {epoch_loss:.4f}")
    print(f"{epoch_prefix} Training f1: {tr_f1:.4f}")
    print(f"{epoch_prefix} Training precision : {tr_p:.4f}")
    print(f"{epoch_prefix} Training recall: {tr_r:.4f}")
    print(f"{epoch_prefix} Model saved to pytorch_model_e{epoch}.bin  [{elapsed / 60:.2f} mins]")
    # wandb.log({'Train Loss (Epoch)': epoch_loss, 'Train Accuracy (Epoch)' : tr_accuracy})
    print(epoch_prefix)


def evaluate(val):
    print(calculate_char_CV(val))


config = {'train_batch_size': 4,
          'valid_batch_size': 2,
          'epochs': 5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm': 10,
          'device': 'cuda' if torch.cuda.is_available() else 'cpu',

          }
net = NeuralNetwork()
optimizer = torch.optim.Adam(params=net.parameters(), lr=config['learning_rates'][0])
import time

folds = np.unique(merged['fold'])
for fold in folds:
    d = merged[merged['fold'] == fold].index
    d = tokenize(d, tensor=True)

    ds = scoreDataset(d)
    ds = DataLoader(ds, batch_size=3, shuffle=True, num_workers=0, pin_memory=True)

    val = scoreDataset(d)
    val = DataLoader(val, batch_size=3, shuffle=False, num_workers=0, pin_memory=True)

    print(len(d['input_ids']))
    for epoch in range(config['epochs']):
        train(net.to(device), optimizer, ds, epoch)
    evaluate(val)
    torch.save(net.state_dict(), f'fold_{fold}.bin')
    print(f" Model saved to pytorch_model_e{fold}.bin]")

all_ = merged.index
all_ = tokenize(all_, tensor=True)
all_ = scoreDataset(all_)
all_ = DataLoader(all_, batch_size=3, shuffle=False, num_workers=0, pin_memory=True)
evaluate(all_)

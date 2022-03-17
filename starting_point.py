"""
This gives a useful stating point. It provides:
    - a benchmark evaluation metric
    - a method for creating folds
    - train evaluate, inference and score a baseline of 0.568 without deep learning
    - this gives a benchmark against which to measure more complex solutions
"""
import os
import ast
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

pd.options.display.max_colwidth = 200

# constants
DATA_PATH = "data"
K = 5
SEED = 2222

# data
patient_notes = pd.read_csv(os.path.join(DATA_PATH, "patient_notes.csv"))
features = pd.read_csv(os.path.join(DATA_PATH, "features.csv"))
df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

#
df_train['annotation'] = df_train['annotation'].apply(ast.literal_eval)
df_train['location'] = df_train['location'].apply(ast.literal_eval)
df_train = df_train.merge(features, how="left", on=["case_num", "feature_num"])

df_train_grouped = df_train.groupby(['case_num', 'pn_num']).agg(list)
patient_notes = patient_notes.merge(df_train_grouped, how="left", on=['case_num', 'pn_num'])
patient_notes = patient_notes.dropna(axis=0).reset_index(drop=True)

patient_notes = patient_notes[[
    'case_num',
    'pn_num',
    'pn_history',
    'annotation',
    'location',
    'feature_text',
    'feature_num'
]]

"""
Folds
There are two possibilities that come to my mind for splitting the data : 
- A k-fold on features stratified by `case_num`
- A k-fold on features grouped by `case_num`

From my understanding, clinical cases will be the same in the train and test data, 
hence I'm going with the first option.
"""
skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)
splits = list(skf.split(X=patient_notes, y=patient_notes['case_num']))
folds = np.zeros(len(patient_notes), dtype=int)
for i, (train_idx, val_idx) in enumerate(splits):
    folds[val_idx] = i
    df_val = patient_notes.iloc[val_idx]
    print(f'   -> Fold {i}')
    print('- Number of samples :', len(df_val))
    print('- Case repartition :', dict(Counter(df_val['case_num'])), '\n')

patient_notes['fold'] = folds
patient_notes[['case_num', 'pn_num', 'fold']].to_csv('folds.csv', index=False)

"""
# Metric

From the [evaluation page](https://www.kaggle.com/c/nbme-score-clinical-patient-notes/overview/evaluation) :
- This competition is evaluated by a micro-averaged F1 score.
- We score each character index as:
 - TP if it is within both a ground-truth and a prediction,
 - FN if it is within a ground-truth but not a prediction, and,
 - FP if it is within a prediction but not a ground truth.
- Finally, we compute an overall F1 score from the TPs, FNs, and FPs aggregated across all instances.
"""


def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.
    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.
    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


preds = [[0, 0, 1], [0, 0, 0]]
truths = [[0, 0, 1], [1, 0, 0]]
micro_f1(preds, truths)


# Now we need to convert predicted spans to binary arrays indcating whether each character is predicted of not.
def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.
    Args: spans (list of lists of two ints): Spans.
    Returns: np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


spans_to_binary([[0, 5], [10, 15]])


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.
    :param preds: (list of lists of two ints): Prediction spans.
    :param truths: (list of lists of two ints): Ground truth spans.
    :return: float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))

    return micro_f1(bin_preds, bin_truths)


# We generate spans from a train example.

spans = patient_notes['location'][0]
spans = [[list(np.array(s.split(' ')).astype(int)) for s in span] for span in spans if len(span)]

pred = spans
truth = [span[:2] for span in spans]

print(pred)
print(truth)

span_micro_f1(pred, truth)

# Baseline: We basically perform string matching on all the data.
# Preparation


def location_to_span(location):
    spans = []
    for loc in location:
        if ";" in loc:
            loc = loc.split(';')
        else:
            loc = [loc]
        for l in loc:
            spans.append(list(np.array(l.split(' ')).astype(int)))
    return spans


df = df_train.copy()
patient_notes = pd.read_csv(os.path.join(DATA_PATH, "patient_notes.csv"))
df = df.merge(patient_notes, how="left")


df_folds = pd.read_csv('folds.csv')
df = df.merge(df_folds, how="left", on=["case_num", "pn_num"])
df['span'] = df['location'].apply(location_to_span)


# Evaluation
for fold in range(K):
    print(f"\n-------------   Fold {fold + 1} / {K}  -------------\n")
    df_train = df[df['fold'] != fold].reset_index(drop=True)
    df_val = df[df['fold'] == fold].reset_index(drop=True)
    matching_dict = df_train[['case_num', 'feature_num', 'annotation']].groupby(['case_num', 'feature_num']).agg(
        list).T.to_dict()
    matching_dict = {k: np.concatenate(v['annotation']) for k, v in matching_dict.items()}
    matching_dict = {k: np.unique([v_.lower() for v_ in v]) for k, v in matching_dict.items()}
    preds = []
    for i in range(len(df_val)):
        key = (df_val['case_num'][i], df_val['feature_num'][i])
        #         print(key)
        candidates = matching_dict[key]
        text = df_val['pn_history'][i].lower()
        spans = []
        for c in candidates:
            start = text.find(c)
            if start > -1:
                spans.append([start, start + len(c)])
        preds.append(spans)
    score = span_micro_f1(preds, df_val['span'])
    print(f"-> F1 score: {score :.3f}")

# Inference
df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
df_test = df_test.merge(patient_notes, how="left")

df_train = df.copy()

matching_dict = df_train[['case_num', 'feature_num', 'annotation']].groupby(['case_num', 'feature_num']).agg(list).T.to_dict()
matching_dict = {k: np.concatenate(v['annotation']) for k, v in matching_dict.items()}
matching_dict = {k: np.unique([v_.lower() for v_ in v]) for k, v in matching_dict.items()}

preds = []
for i in range(len(df_test)):
    key = (df_test['case_num'][i], df_test['feature_num'][i])
    candidates = matching_dict[key]
    text = df_test['pn_history'][i].lower()
    spans = []
    for c in candidates:
        start = text.find(c)
        if start > -1:
            spans.append([start, start + len(c)])
    preds.append(spans)


# Submission
def preds_to_location(preds):
    locations = []
    for pred in preds:
        loc = ";".join([" ".join(np.array(p).astype(str)) for p in pred])
        locations.append(loc)
    return locations


sub = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
sub['location'] = preds_to_location(preds)
sub.to_csv('submission.csv', index=False)


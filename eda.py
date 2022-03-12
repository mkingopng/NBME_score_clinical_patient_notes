"""

"""
from config import *


print(feats_df.head())

# 131/143 of the features are unique and it looks like some have OR delimiting multiple names
len(feats_df), feats_df.feature_text.nunique()

print(notes_df.head())
print("DataFrame shape", notes_df.shape)
print("Unique case_num values", notes_df.case_num.unique())
print("Number of unique pn_num", notes_df.pn_num.nunique())
print("Number of unique note texts", notes_df.pn_history.nunique())

# Somewhat unequal distribution of notes for each case

px.histogram(notes_df, x="case_num", color="case_num")

print(train_df.shape)
print(train_df.head())

# Checking for missing annotations
missing_annotations = train_df["annotation"]=="[]"
missing_locations = train_df["location"]=="[]"
both_missing = (train_df["annotation"] == train_df["location"])&missing_annotations

sum(missing_annotations), sum(missing_locations), sum(both_missing)
# About 4.4k rows missing.

# Looking at distribution of case numbers in train.csv
px.histogram(train_df, x="case_num", color="case_num")

# Looking at distribution of patient note numbers in train.csv
px.histogram(train_df, x="pn_num", color="case_num")

# Equal numbers of features for each case_num
px.histogram(train_df, x="feature_num", color="case_num", nbins=1000)
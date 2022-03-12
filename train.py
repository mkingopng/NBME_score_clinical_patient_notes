"""

"""
from config import *
from functions import *
from wandb_creds import *

# data cleanup: The annotation and location columns are loaded as strings. This turns them back into lists.
train_df["anno_list"] = [literal_eval(x) for x in train_df.annotation]
train_df["loc_list"] = [literal_eval(x) for x in train_df.location]
train_df.head()

# stratified kfold
skf = StratifiedKFold(n_splits=data_args.k_folds, random_state=training_args.seed, shuffle=True)

splits = list(skf.split(X=notes_df, y=notes_df['case_num']))

notes_df["fold"] = -1

for fold, (_, val_idx) in enumerate(skf.split(notes_df, y=notes_df["case_num"])):
    notes_df.loc[val_idx, "fold"] = fold

counts = notes_df.groupby(["fold", "pn_num"], as_index=False).count()

# If the number of rows is the same as the number of unique pn_num, then each pn_num is only in one fold. Also if all
# the counts=1
print(counts.shape, counts.pn_num.nunique(), counts.case_num.unique())
print(counts)

merged = train_df.merge(notes_df, how="left")
merged = merged.merge(feats_df, how="left")
print(merged.head(10))

# Correcting some incorrect annotations
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

merged.loc[3858, "anno_list"] = '["previously as regular", "previously eveyr 28-29 days", "previously lasting 5 days", "previously regular flow"]'
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

merged.loc[6380, "anno_list"] = '["No hair changes", "No skin changes", "No GI changes", "No palpitations", "No excessive sweating"]'
merged.loc[6380, "loc_list"] = '["480 482;507 519", "480 482;499 503;512 519", "480 482;521 531", "480 482;533 545", "480 482;564 582"]'

merged.loc[6562, "anno_list"] = '["stressed due to taking care of her mother", "stressed due to taking care of husbands parents"]'
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

merged.loc[9938, "anno_list"] = '["irregularity with her cycles", "heavier bleeding", "changes her pad every couple hours"]'
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

merged.loc[12279, "anno_list"] = '["heard what she described as a party later than evening these things did not actually happen"]'
merged.loc[12279, "loc_list"] = '["405 459;488 524"]'

merged.loc[12289, "anno_list"] = '["experienced seeing her son at the kitchen table these things did not actually happen"]'
merged.loc[12289, "loc_list"] = '["353 400;488 524"]'

merged.loc[13238, "anno_list"] = '["SCRACHY THROAT", "RUNNY NOSE"]'
merged.loc[13238, "loc_list"] = '["293 307", "321 331"]'

merged.loc[13297, "anno_list"] = '["without improvement when taking tylenol", "without improvement when taking ibuprofen"]'
merged.loc[13297, "loc_list"] = '["182 221", "182 213;225 234"]'

merged.loc[13299, "anno_list"] = '["yesterday", "yesterday"]'
merged.loc[13299, "loc_list"] = '["79 88", "409 418"]'

merged.loc[13845, "anno_list"] = '["headache global", "headache throughout her head"]'
merged.loc[13845, "loc_list"] = '["86 94;230 236", "86 94;237 256"]'

merged.loc[14083, "anno_list"] = '["headache generalized in her head"]'
merged.loc[14083, "loc_list"] = '["56 64;156 179"]'

merged["anno_list"] = [literal_eval(x) if isinstance(x, str) else x for x in merged["anno_list"]]
merged["loc_list"] = [literal_eval(x) if isinstance(x, str) else x for x in merged["loc_list"]]

merged = merged[merged["anno_list"].map(len) != 0].copy().reset_index(drop=True)

print(merged.head())

# Tokenizing and Adding Labels: Since the labeling is given to us at the character level, the tokenizer needs to have
# `return_offsets_mapping=True` which returns the start and end indexes for each token. These indexes can then map the
# char-level labels to tokens. The loss for the model must be calculated at the token level. Here are the 3 scenarios
# where I mark a token as a label.

# `token_start, token_end` are the start and end indexes of the token. start is inclusive, end is exclusive, just like
# indexing a string.

# `label_start, label_end` are the start and end indexes of the label. start is inclusive, end is exclusive, just like
# indexing a string.

# 1. `token_start <= label_start < token_end`
# The token span overlaps with the start of the label span.

# 2. `token_start < label_end <= token_end`
# The token span overlaps with the end of the label span.

# 3. `label_start <= token_start < label_end`
# If it doesn't fall into (1) or (2), then the token span is entirely in the label span.

merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]

# Double-checking alignment is good
if "deberta-v2" in model_args.model_name_or_path or "deberta-v3" in model_args.model_name_or_path:
    from transformers.models.deberta_v2 import DebertaV2TokenizerFast

    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_args.model_name_or_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

first = merged.sample(n=1).iloc[0]
example = {
    "feature_text": first.feature_text,
    "text": first.pn_history,
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

dataset = Dataset.from_pandas(
    merged[["id", "case_num", "pn_num", "feature_num", "loc_list", "pn_history", "feature_text", "fold"]])

dataset = dataset.rename_column("pn_history", "text")
if DEBUG:
    dataset = dataset.shuffle().select(range(1000))

# This can take up to a minute
tokenized_dataset = dataset.map(
    partial(
        tokenize_and_add_labels,
        tokenizer=tokenizer),
    desc="Tokenizing and adding labels",
    num_proc=4
)

print(tokenized_dataset)

print(tokenized_dataset[0])

# How long are the texts?
tokenized_lengths = [len(x) for x in tokenized_dataset["input_ids"]]

print("The longest is", max(tokenized_lengths))

px.histogram(x=tokenized_lengths, labels={"x":"tokenized_length"})

# Setup Weights and Biases for tracking experiments
# If you put report_to="none" in the TrainingArguments then it won’t use Weights and Biases.
# I like using it because it helps keep track of experiments.

wandb.login(key=API_KEY)

# Model backbone flexibility
# This custom model is a bit funky because I tried to make it versatile to whichever model you would like to use (bert,
# roberta, electra, etc.). It works by pulling the proper classes based on the model_type specified in the config
# object. If you know of a better way, by all means please share! Unfortunately there are minor differences in how each
# model is set up, so there are exceptions here and there for individual models.

if __name__ == "__main__":
    if DEBUG:
        training_args.num_train_epochs = 1

    previous_config = None
    for fold, model_name in zip(range(data_args.k_folds), all_models):

        """
        This seems to get reset after each fold and can print out a lot of 
        information that I don't really care about. When debugging, you should
        definitely not hide these messages though 😉
        """
        if not DEBUG:
            logging.set_verbosity(logging.CRITICAL)

        print(f"Starting training for fold {fold} using {model_name}")

        config = AutoConfig.from_pretrained(
            model_name,
        )
        using_deberta_v2_3 = "deberta-v2" in model_name or "deberta-v3" in model_name

        # Only re-run when the config changes
        if previous_config is None or previous_config != config.__dict__:

            if using_deberta_v2_3:
                from transformers.models.deberta_v2 import DebertaV2TokenizerFast

                tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)

            data_collator = CustomDataCollator(
                tokenizer,
                pad_to_multiple_of=8 if training_args.fp16 else None,
                padding=True,
                max_length=data_args.max_seq_length,
            )

            tokenized_dataset = dataset.map(
                partial(
                    tokenize_and_add_labels,
                    tokenizer=tokenizer,
                ),
                desc="Tokenizing and adding labels",
                num_proc=4)

        model_args.model_name_or_path = model_name  # So wandb will track it

        if "wandb" in training_args.report_to:
            wandb_config = {**model_args.__dict__, **data_args.__dict__, **training_args.__dict__, **config.__dict__,
                            "fold": fold}
            wandb.init(config=wandb_config, group=os.environ["WANDB_RUN_GROUP"])

        model = get_model(model_name, config)

        # Initialize our Trainer
        trainer = NBMETrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset.filter(lambda x: x["fold"] != fold),
            eval_dataset=tokenized_dataset.filter(lambda x: x["fold"] == fold),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(f"fold{fold}")

        if "wandb" in training_args.report_to:
            wandb.finish()

        previous_config = config.__dict__
        
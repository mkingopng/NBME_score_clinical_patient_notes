
# 1st place

## Bert token prediction

## LGB sentence prediction 
ref chase bowers again

## model ensemble

# 2nd Place
## models:
- deberta-large
- deberta-v3-large
- deberta-xl
- deberta-v2-xl
- longformer-large-4096
- LSG converted roberta
- funnel transformer/largge
- bigbird-roberta-base
- yoso-4096

## 60% data training oof

## weighted box fusion approach to ensemble

## Post process
- repairing span predictions
- discourse-specific rules
- adjusting lengths of predicted spans

# 3rd Place:
## Models
- 3.1: longformer, deberta-l, SW deberta-xl
- 3.2: augmentation - masked augmentation and cutmix
- 3.3: hyperparameters

## stacking Framework
as per chase bowers but improved.
- CV setup
- features
- increasing the amount of candidate spans
- gradient boosting models accelerated with RAPIDS
- decoding

# My strategy?

## models
- Bert-large: 
- deberta-large: improves BERT and RoBERTa using disentangled attention and enhanced mask encoder
- deberta-v2: improvements
- deberta-v3: further improvements
- T5: 
- Roberta-large: pretrained using masked langauge modeling

## optimize hypers for each model & train model
- use WandB sweeps to optimise HPs
- consider using AWS for training & optimisation once the script is working
- need to learn how to use docker to make AWS viable

## ensembling method
- what do you think?

## LGB for optimising ... something?
- 
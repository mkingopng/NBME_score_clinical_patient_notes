
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

## simplify my structure & use the API
- do what 2nd place did and mirror Kaggle's directory structure locally & in docker
- use the Kaggle API to upload datasets and submissions
- use Kaggle API to use all possible tries each day: 5 attempts each day
- use code to record the outcomes. maybe in a simple DB?
- strategy for submissions?

## resources:
No matter models, ensemble method and post-processing I choose below, I need to fine-tune up as many good models as 
I can, right? Is it a matter of setting up a training script like i have and then repeating it many times to get the 
best fine-tuned checkpoints for each model selected?

If that's the case then I guess it's a matter of running both my computers full-time training, and learn how to move 
some workload to AWS. I can get EC2 G4 instances from a couple of bucks an hour. I need to learn how to use docker, how 
to use multi-gpu, and how to run code on AWS.

## models -> on what basis am I selecting models?


- Bert: 
- deberta-large: improves BERT and RoBERTa using disentangled attention and enhanced mask encoder
- deberta-v2: improvements
- deberta-v3: further improvements
- T5: 
- Roberta-large: pretrained using masked langauge modeling

## optimize HPs for each model & train model
- use WandB sweeps to optimise HPs
- run both GPUs constantly
- consider using AWS for training & optimisation once the script is working
- need to learn how to use docker to make AWS viable

## ensemble method
- weighted box fusion approach?
- 
- what do you think?

## LGB for optimising ... something?
- 
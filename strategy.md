
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

############################################################################

# My strategy?
-> What is the problem? NER & Q&A (i think)
-> Iterate through the top models & naive ensemble

## simplify my structure & use the Kaggle API
- do what 2nd place did and mirror Kaggle's directory structure locally & in docker
- use the Kaggle API to upload datasets and submissions
- use Kaggle API to use all possible tries each day: 5 attempts each day
- use code to record the outcomes. maybe in a simple DB?
- strategy for submissions?

## resources:
No matter the models, ensemble method and post-processing I choose below, I need to fine-tune up as many good models as 
I can, right? Is it a matter of setting up a training script like i have and then repeating it many times to get the 
best fine-tuned checkpoints for each model selected?

If that's the case then I guess it's a matter of running both my computers full-time training, and learn how to move 
some workload to AWS. I can use sagemaker to run the training on the cloud.
- Use AWS sagemaker to run the training
- Need to learn how to do it!
- I get 2 months free when i start. after that its between 0.2 - 0.6 per hour for 16 - 64GB on demand instances
- https://aws.amazon.com/sagemaker/pricing/
- HuggingFace has resources to teach you how to run training on sagemaker
- https://huggingface.co/docs/transformers/sagemaker

## Data
- missing data & errors -> text generation?
- some are not tagged or labelled
- data augmentation: nlpaug?

## feature engineering

## preprocessing
- preprocessing is required

## post-processing
- post-processing is required

## models -> on what basis am I selecting models?
- BERT: 
- deberta-large: improves BERT and RoBERTa using disentangled attention and enhanced mask encoder
- deberta-v2: improvements
- deberta-v3: further improvements
- T5: 
- Roberta-large: pretrained using masked langauge modeling
- distilbert: 

# Model Notes
- I don't need both the pretrained model and the fine-tuned checkpoints for inference. 
- I just need to the pretrained model and tokenizer files for inference
- need to score each model individually on kaggle
- need to build a scaffold to fine-tune each model, and to perform inference
- need enough diversity in models

## optimize HPs for each model & train model
- use WandB sweeps to optimise HPs
- run both GPUs constantly
- consider using AWS for training & optimisation once the script is working
- need to learn how to use sagemaker/docker to make AWS viable

## Tokenizer
-> tokenizer files are generated when the tokenizer is trained on the corpus
- special_tokens_map.json
- tokenizer.json

## ensemble method
- looks like 1st place is basically a naive approach. read the code
- 2nd place weighted box fusion approach? what is this?
- what is the chase bowers approach referred to in 3rd place?
- don't do the bayesian thing

## LGB for optimising ... something?
- optimizing the prediction threshold?
- optimise the ensemble: accelerate the data frame operations using cuDF & LGB????
- What is the special technique that Giba uses? Send podcast to wilson

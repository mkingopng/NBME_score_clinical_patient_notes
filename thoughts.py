"""
Some feedback and thoughts as i start to wind up on this project.
"""
# mk: there are lots of spelling mistakes and other errors

# mk: there are lots of technical terms that are unlikely to be covered in normal language models.
#  use a model trained on medical corpus?

# mk:the average text length is quite short so a normal length language model should suffice

# mk: lots of missing annotations, imbalanced classes, a good incentive to use some synthetic data?

# mk: I don't think this is a straight NER task. It has elements of question/answer or fill-mask.
#  I can see that other competitors have approached it from a hybrid approach for this reason i think

# mk:batch_size=8 is clearly a sweet spot in terms of speed. Increasing to 12 does little to increase speed but accuracy
#  suffers. Dropping to 4 improves accuracy a little but training time more than doubles.

# mk: Batch_number-4 improves CV but more than doubles train time. save this for once other parameters are finalized.
#  Increasing > 8 doesn't reduce time much but reduces accuracy. 8 is the sweet spot for tuning.

# mk: n_folds=10 seems ot improve CV but increases training time. leave it at 5 until other hypers are tuned

# mk: n_epochs =  10 seems to improve CV but increases training time. Leave it at 5 until other hypers are tuned

# mk: decoder_lr:
#  2e-5:
#  3e-5:
#  4e-5:
#  5e-5:

# mk: encoder_lr =

# mk: min_lr =

# mk: models to ensemble
#  - deberta-base
#  - deberta-large
#  - deberta-v2-large
#  - deberta-v3-large
#  - roberta

# todo: need to clearly define the problem asap. If its not NER, what is it? This will inform the choice of model and
#  the structure.

# mk: i think its a combination of NER & Q/A

# todo: can i test the coverage of the tokenizer like that notebook in FP?

# todo: get the program working using the huggingface hub for model and tokenizer source/storage. Faster to test
#  different models & tokenizersSave locally once i need to submit on Kaggle.

# todo: test different model and tokenizer alternatives

# todo: need to keep in mind the need for multiple levels of optimization from the beginning:
#  - data engineering & feature engineering
#  - choice of the right pretrained model and tokenizer
#  - training the best fine tuned model
#  - bayesian optimization of hyperparameters
#  - bayesian optimisation of model ensembles.

# todo: Do submissions using Kaggle API
#  - use all 5 submissions per day
#  - start stacking/ensembling early
#  - test different hyperparameters
#  -
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

# todo: need to clearly define the problem asap. If its not NER, what is it? This will inform the choice of model and
#  the structure

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

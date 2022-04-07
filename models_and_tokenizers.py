from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead, AutoModelForCausalLM, \
    RobertaTokenizerFast

# heading: bert-base-uncased
bert_base_uncased_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_base_uncased_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

bert_base_uncased_model.save_pretrained("kaggle/input/models/bert_base_uncased_model")
bert_base_uncased_tokenizer.save_pretrained("kaggle/input/tokenizers/bert_base_uncased_tokenizer")

# git clone https://huggingface.co/bert-base-uncased

# when offline
# bert_base_uncased_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/bert_base_model")
# bert_base_uncased_model = AutoModel.from_pretrained("kaggle/input/tokenizers/bert_base_tokenizer")

# heading: roberta-base
roberta_base_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_fast_base_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
roberta_base_model = AutoModelForMaskedLM.from_pretrained("roberta-base")

roberta_base_model.save_pretrained("kaggle/input/models/roberta_base_model")
roberta_base_tokenizer.save_pretrained("kaggle/input/tokenizers/roberta_base_tokenizer")
roberta_fast_base_tokenizer.save_pretrained("kaggle/input/tokenizers/roberta_fast_base_tokenizer")
# git clone https://huggingface.co/roberta-base

# when offline
# roberta_base_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/roberta_base_model")
# roberta_base_model = AutoModel.from_pretrained("kaggle/input/tokenizers/roberta_base_tokenizer")
# roberta_fast_base_tokenizer = RobertaTokenizerFast.from_pretrained("kaggle/input/tokenizers/roberta_fast_base_tokenizer")

# heading: roberta-large
roberta_large_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
roberta_large_model = AutoModelForMaskedLM.from_pretrained("roberta-large")
roberta_fast_large_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

roberta_large_model.save_pretrained("kaggle/input/models/roberta_large_model")
roberta_large_tokenizer.save_pretrained("kaggle/input/tokenizers/roberta_large_tokenizer")
roberta_fast_base_tokenizer.save_pretrained("kaggle/input/tokenizers/roberta_fast_base_tokenizer")
# git clone https://huggingface.co/roberta-large

# when offline
# roberta_large_model = AutoModel.from_pretrained("kaggle/input/tokenizers/roberta_large_tokenizer")
# roberta_large_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/roberta_large_model")
# roberta_fast_base_tokenizer.save_pretrained("kaggle/input/tokenizers/roberta_fast_large_tokenizer")

# heading: t5-base
t5_base_tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_base_model = AutoModelWithLMHead.from_pretrained("t5-base")

t5_base_model.save_pretrained("kaggle/input/models/t5_base_model")
t5_base_tokenizer.save_pretrained("kaggle/input/tokenizers/t5_base_tokenizer")
# git clone https://huggingface.co/t5-base

# when offline
# t5_base_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/t5_base_model")
# t5_base_model = AutoModel.from_pretrained("kaggle/input/tokenizers/t5_base_tokenizer")

# heading: t5-large
t5_large_tokenizer = AutoTokenizer.from_pretrained("t5-large")
t5_large_model = AutoModelWithLMHead.from_pretrained("t5-large")

t5_large_model.save_pretrained("kaggle/input/models/t5_large_model")
t5_large_tokenizer.save_pretrained("kaggle/input/tokenizers/t5_large_tokenizer")
# git clone https://huggingface.co/t5-large

# when offline
t5_large_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/t5_large_model")
t5_large_base_model = AutoModel.from_pretrained("kaggle/input/tokenizers/t5_large_tokenizer")

# heading: distilgpt2
distilgpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
distilgpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

distilgpt2_model.save_pretrained("kaggle/input/models/distilgpt2_model")
distilgpt2_tokenizer.save_pretrained("kaggle/input/tokenizers/distilgpt2_tokenizer")
# git clone https://huggingface.co/distilgpt2

# when offline
distilgpt2_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/roberta_base_model")
distilgpt_model = AutoModel.from_pretrained("kaggle/input/tokenizers/roberta_base_tokenizer")

# heading: deberta-base
# tokenizer = AutoTokenizer.from_pretrained("deberta-base")  # fix_me
# model = AutoModelForMaskedLM.from_pretrained("deberta-base")
# model.save_pretrained("kaggle/input/models/deberta_base_model")
# tokenizer.save_pretrained("kaggle/input/tokenizers/deberta_base_tokenizer")
# git clone https://huggingface.co/bert-base-uncased

# when offline
# deberta_base_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/deberta_base_model")
# deberta_base_model = AutoModel.from_pretrained("kaggle/input/tokenizers/deberta_base_tokenizer")

# heading: deberta-large
# tokenizer = AutoTokenizer.from_pretrained("deberta-large")  # fix_me
# model = AutoModelForMaskedLM.from_pretrained("deberta-large")
# model.save_pretrained("kaggle/input/models/deberta_large_model")
# tokenizer.save_pretrained("kaggle/input/tokenizers/deberta_large_tokenizer")
# git clone

# when offline
# deberta_large_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/deberta_large_model")
# deberta_large_model = AutoModel.from_pretrained("kaggle/input/tokenizers/deberta_large_tokenizer")

# heading: deberta-v2-xlarge
# tokenizer = AutoTokenizer.from_pretrained("deberta-v2-xlarge")  # fix_me
# model = AutoModelForMaskedLM.from_pretrained("deberta-v2-xlarge")
# model.save_pretrained("kaggle/input/models/deberta_v2_xlarge_model")
# tokenizer.save_pretrained("kaggle/input/tokenizers/deberta_v2_xlarge_tokenizer")
# git clone

# when offline
# deberta_v2_xlarge_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/deberta_v2_xlarge_model")
# deberta_v2_xlarge_model = AutoModel.from_pretrained("kaggle/input/tokenizers/deberta_v2_xlarge_tokenizer")

# heading: deberta-v3-large
# tokenizer = AutoTokenizer.from_pretrained("deberta-v3-large")  # fix_me
# model = AutoModelForMaskedLM.from_pretrained("deberta-v3-large")
# model.save_pretrained("kaggle/input/models/deberta_v3_large_model")
# tokenizer.save_pretrained("kaggle/input/tokenizers/deberta_v3_large_tokenizer")
# git clone

# when offline
# deberta_v3_large_tokenizer = AutoTokenizer.from_pretrained("kaggle/input/models/deberta_v3_large_model")
# deberta_v3_large_model = AutoModel.from_pretrained("kaggle/input/tokenizers/deberta_v3_large_tokenizer")
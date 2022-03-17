"""

"""
from config import *


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


def tokenize_and_add_labels(example, tokenizer):
    tokenized_inputs = tokenizer(
        example["feature_text"],
        example["text"],
        truncation="only_second",
        max_length=data_args.max_seq_length,
        padding=True,
        return_offsets_mapping=True
    )

    # labels should be float
    labels = [0.0] * len(tokenized_inputs["input_ids"])
    tokenized_inputs["locations"] = loc_list_to_ints(example["loc_list"])
    tokenized_inputs["sequence_ids"] = tokenized_inputs.sequence_ids()

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


# Functions that are similar across all models
def __init__(self, config):
    super(self.PreTrainedModel, self).__init__(config)

    kwargs = {
        "add_pooling_layer": False
    }
    if config.model_type not in {"bert", "roberta"}:
        kwargs = {}
    setattr(self, self.backbone_name, self.ModelClass(config, **kwargs))

    classifier_dropout_name = None
    for key in dir(config):
        if ("classifier" in key or "hidden" in key) and "dropout" in key:
            if getattr(config, key) is not None:
                classifier_dropout_name = key
                break

    if classifier_dropout_name is None:
        raise ValueError("Cannot infer dropout name in config")
    classifier_dropout = getattr(config, classifier_dropout_name)
    self.dropout = torch.nn.Dropout(classifier_dropout)
    self.classifier = torch.nn.Linear(config.hidden_size, 1)


def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Funky alert
    outputs = getattr(self, self.backbone_name)(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        # head_mask=head_mask, # these aren't necessary and some models error if you include
        # inputs_embeds=inputs_embeds,  # these aren't necessary and some models error if you include
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]

    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))

        # this ignores the part of the sequence that got -100 as labels
        loss = torch.masked_select(loss, labels.view(-1, 1) > -1).mean()

    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@classmethod
def from_pretrained(cls, *args, **kwargs):
    cls.PreTrainedModel = kwargs.pop("PreTrainedModel")
    cls.ModelClass = kwargs.pop("ModelClass")
    cls.backbone_name = kwargs["config"].model_type

    # changes deberta-v2 --> deberta
    if "deberta" in cls.backbone_name:
        cls.backbone_name = "deberta"

    return super(cls.PreTrainedModel, cls).from_pretrained(*args, **kwargs)


def get_model(model_name_or_path, config):
    model_type = type(config).__name__[:-len("config")]
    name = f"{model_type}PreTrainedModel"
    PreTrainedModel = getattr(__import__("transformers", fromlist=[name]), name)
    name = f"{model_type}Model"
    ModelClass = getattr(__import__("transformers", fromlist=[name]), name)

    model = type('HybridModel',
                 (PreTrainedModel,),
                 {'__init__': __init__,
                  "forward": forward,
                  "from_pretrained": from_pretrained
                  }
                 )

    model._keys_to_ignore_on_load_unexpected = [r"pooler"]
    model._keys_to_ignore_on_load_missing = [r"position_ids"]

    return model.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                 PreTrainedModel=PreTrainedModel,
                                 ModelClass=ModelClass,
                                 config=config)


@dataclass
class CustomDataCollator(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Have to modify to make label tensors float and not int.
    """
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100
    return_tensors = "pt"

    def torch_call(self, features):
        batch = super().torch_call(features)
        label_name = "label" if "label" in features[0].keys() else "labels"
        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float32)
        return batch


# Getting locations based on predictions: For each token if the value after a logit goes through a sigmoid is > 0.5,
# then it is an important token. This is a simple approach, and it would be good to test out different numbers in CV.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_metrics(eval_prediction):
    """
    This only gets the scores at the token level. The actual leaderboard score is based at the character level.
    The CV score at the character level is handled in the evaluate function of the trainer.
    """
    predictions, y_true = eval_prediction
    predictions = sigmoid(predictions)
    y_true = y_true.astype(int)

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

        all_predictions.append(current_preds)

    return all_predictions


# Calculate a char-level CV score
# I include precision and recall in addition to f1 to get a better sense of the types of errors the model is making.

def calculate_char_CV(dataset, predictions):
    """
    Some tokenizers include the leading space as the start of the offset_mapping, so there is code to ignore that space.
    """
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels, text in zip(
            predictions,
            dataset["offset_mapping"],
            dataset["sequence_ids"],
            dataset["labels"],
            dataset["text"]
    ):

        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros((num_chars))

        for o, s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:  # ignore question part of input
                continue
            if int(label) == 1:

                char_labels[o[0]:o[1]] = 1
                if text[o[0]].isspace() and o[0] > 0 and char_labels[o[0] - 1] != 1:
                    char_labels[o[0]] = 0

        char_preds = np.zeros((num_chars))

        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1
            if text[start_idx].isspace():
                char_preds[start_idx] = 0

        all_labels.extend(char_labels)
        all_preds.extend(char_preds)

    results = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2]
    }


class NBMETrainer(Trainer):
    """
    I override the default evaluation loop to include
    a character-level CV. The compute_metrics does the
    scoring at the token level, but the leaderboard score
    is at the character level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The Trainer hides some columns that are necessary for CV
        # This code makes those columns accessible
        dataset_type = kwargs["eval_dataset"].format["type"]
        dataset_columns = list(kwargs["eval_dataset"].features.keys())
        self.cv_dataset = kwargs["eval_dataset"].with_format(type=dataset_type, columns=dataset_columns)

    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only=None,
            ignore_keys=None,
            metric_key_prefix="eval",
    ):

        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )

        # This same loop gets called during predict, and we can't do CV when predicting
        is_in_eval = metric_key_prefix == "eval"

        # Custom CV F1 calculation
        if is_in_eval:
            eval_preds = get_location_predictions(self.cv_dataset, eval_output.predictions)

            char_scores = calculate_char_CV(self.cv_dataset, eval_preds)

            for name, score in char_scores.items():
                eval_output.metrics[f"{metric_key_prefix}_char_{name}"] = score

        return eval_output

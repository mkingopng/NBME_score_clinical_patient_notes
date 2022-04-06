"""
Libraries
"""
from IPython.core.display_functions import display
from functions import *


"""
Configuration
"""


class CONFIGURATION:
    num_workers = 4
    path = "../input/nbme-deberta-large-checkpoints/"  # todo: adjust this for each scoring attempt
    config_path = path + 'config.pth'
    model = "microsoft/deberta-large"
    batch_size = 24
    fc_dropout = 0.2
    max_len = 466
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]


"""
settings
"""
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
TOKENIZERS_PARALLELISM = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_everything(seed=42)

"""
oof
"""
oof = pd.read_pickle(CONFIGURATION.path + 'oof_df.pkl')

truths = create_labels_for_scoring(oof)
char_probs = get_char_probs(oof['pn_history'].values,
                            oof[[i for i in range(CONFIGURATION.max_len)]].values,
                            CONFIGURATION.tokenizer)
best_th = 0.5
best_score = 0.
for th in np.arange(0.45, 0.55, 0.01):
    th = np.round(th, 2)
    results = get_results(char_probs, th=th)
    preds = get_predictions(results)
    score = get_score(preds, truths)
    if best_score < score:
        best_th = th
        best_score = score
    LOGGER.info(f"th: {th}  score: {score:.5f}")
LOGGER.info(f"best_th: {best_th}  score: {best_score:.5f}")

"""
Test Data Loading
"""

test = pd.read_csv('../input/nbme-score-clinical-patient-notes/test.csv')
submission = pd.read_csv('../input/nbme-score-clinical-patient-notes/sample_submission.csv')
features = pd.read_csv('../input/nbme-score-clinical-patient-notes/features.csv')


def preprocess_features(features):
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    return features


features = preprocess_features(features)
patient_notes = pd.read_csv('../input/nbme-score-clinical-patient-notes/patient_notes.csv')
print(f"test.shape: {test.shape}")
display(test.head())
print(f"features.shape: {features.shape}")
display(features.head())
print(f"patient_notes.shape: {patient_notes.shape}")
display(test.head())
test = test.merge(features, on=['feature_num', 'case_num'], how='left')
test = test.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
display(test.head())

"""
Dataset
"""


def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text,
                           add_special_tokens=True,
                           max_length=CONFIGURATION.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.pn_historys[item],
                               self.feature_texts[item])
        return inputs


"""
Model
"""


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


"""
inference
"""


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


test_dataset = TestDataset(CONFIGURATION, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CONFIGURATION.batch_size,
                         shuffle=False,
                         num_workers=CONFIGURATION.num_workers, pin_memory=True, drop_last=False)
predictions = []

for fold in CONFIGURATION.trn_fold:
    model = CustomModel(CONFIGURATION, config_path=CONFIGURATION.config_path, pretrained=False)
    state = torch.load(CONFIGURATION.path + f"{CONFIGURATION.model.replace('/', '-')}_fold{fold}_best.pth",
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    prediction = prediction.reshape((len(test), CONFIGURATION.max_len))
    char_probs = get_char_probs(test['pn_history'].values, prediction, CONFIGURATION.tokenizer)
    predictions.append(char_probs)
    del model, state, prediction, char_probs
    gc.collect()
    torch.cuda.empty_cache()
predictions = np.mean(predictions, axis=0)


"""
submission
"""
results = get_results(predictions, th=best_th)
submission['location'] = results
display(submission.head())
submission[['id', 'location']].to_csv('submission.csv', index=False)

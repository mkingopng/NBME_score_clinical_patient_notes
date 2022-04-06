"""
had 1 bug
- data type error. runs fine on kaggle
- the solution was a bit unusual for me...

found a similar issue
https://github.com/huggingface/transformers/issues/14375
https://huggingface.co/microsoft/deberta-v3-large#fine-tuning-with-hf-transformers

error message was
Traceback (most recent call last): File "/home/noone/Documents/GitHub/NBME_score_clinical_patient_notes
/train.py", line 24, in <module> _oof_df = train_loop(train, fold) File

"/home/noone/Documents/GitHub/NBME_score_clinical_patient_notes/functions.py", line 858, in train_loop
avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

"/home/noone/Documents/GitHub/NBME_score_clinical_patient_notes/functions.py", line 657, in train_fn
scaler.scale(loss).backward()

"/home/noone/anaconda3/envs/NBME/lib/python3.9/site-packages/torch/_tensor.py", line 363, in backward
torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs) File

"/home/noone/anaconda3/envs/NBME/lib/python3.9/site-packages/torch/autograd/__init__.py", line 173, in backward
Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass File

"/home/noone/anaconda3/envs/NBME/lib/python3.9/site-packages/torch/autograd/function.py", line 253, in apply return
user_fn(self, *args) File "/home/noone/anaconda3/envs/NBME/lib/python3.9/site-packages/transformers/models/deberta
/modeling_deberta.py", line 113, in backward inputGrad = _softmax_backward_data(grad_output, output, self.dim,
output) TypeError: _softmax_backward_data(): argument 'input_dtype' (position 4) must be torch.dtype, not Tensor

# fix_me: the error is here. changed File "/home/noone/anaconda3/envs/NBME/lib/python3.9/site-packages/transformers/
   models/deberta/modeling_deberta.py", line 113, in backward inputGrad = _softmax_backward_data(grad_output, output,
   self.dim, output) to inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype) and it fixed
   the issue

"""
from functions import *

if __name__ == '__main__':

    def get_result(oof_df):
        labels = create_labels_for_scoring(oof_df)
        predictions = oof_df[[i for i in range(CONFIGURATION.max_len)]].values
        char_probs = get_char_probs(oof_df['pn_history'].values, predictions, CONFIGURATION.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')


    if CONFIGURATION.train:
        oof_df = pd.DataFrame()
        for fold in range(CONFIGURATION.n_fold):
            if fold in CONFIGURATION.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR + 'oof_df.pkl')

    if CONFIGURATION.wandb:
        wandb.finish()

    gc.collect()
    torch.cuda.empty_cache()
    # wandb.agent( sweep_id, train, count=5)  # todo: necessary for sweeps

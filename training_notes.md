# Summary
- run 1: CV 0.8608, batch 12, 5 epochs, 5 folds, 2h 9m 51s
- run 2: CV 0.8608, batch 12, 5 epochs, 5 folds, 2h 9m 32s
- run 3: CV 0.8555, batch 16, 5 epochs, 5 folds, 2h 4m 59s
- run 4: CV 0.8520, batch 20, 5 epochs, 5 folds, 2h 2m 57s
- run 5: batch 16, 5 epochs, 5 folds
- run 6: batch 20, 5 epochs, 5 folds
- run 7: batch 8, 5 epochs, 5 folds
- run 8: batch 8, 5 epochs, 5 folds
- run 9: batch 4, 
- run 10: batch 4,
- run 11:
- run 12:
- run 13:
- run 14:
- run 15:
- run 16:
- run 17:
- run 18: 
- run 19:


# run 1:
========== CV ==========
Score: 0.8608
wandb: Waiting for W&B process to finish... (success).
wandb: - 0.001 MB of 0.001 MB uploaded (0.000 MB deduped)

wandb: Run history:
wandb: [fold0] avg_train_loss █▂▂▁▁
wandb:   [fold0] avg_val_loss █▃▂▁▂
wandb:          [fold0] epoch ▁▃▅▆█
wandb:           [fold0] loss █▄▃▃▃▃▂▂▃▁▂▁▂▂▄▂▂▂▁▂▂▂▁▁▁▂▃▁▁▁▂▂▂▁▁▂▁▄▁▁
wandb:             [fold0] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold0] score ▁▆▆██
wandb: [fold1] avg_train_loss █▂▁▁▁
wandb:   [fold1] avg_val_loss █▂▂▁▂
wandb:          [fold1] epoch ▁▃▅▆█
wandb:           [fold1] loss ▄▂▃█▆▃▇▃▂▃▄▃▃▄▁▂▂▂▂▂▁▁▂▂▂▂▃▂▂▁▂▃▄▆▃▂▁▂▁▂
wandb:             [fold1] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold1] score ▁▅▇▇█
wandb: [fold2] avg_train_loss █▂▁▁▁
wandb:   [fold2] avg_val_loss █▃▁▁▁
wandb:          [fold2] epoch ▁▃▅▆█
wandb:           [fold2] loss ▆▆▃▆▄▃▅▇▃▃▃▆▂▂▂▂▂▃▂▂▁▁▁▃▁▂▃█▂▂▂▂▂▃▁▂▃▃▁▁
wandb:             [fold2] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold2] score ▁▆▇██
wandb: [fold3] avg_train_loss █▂▁▁▁
wandb:   [fold3] avg_val_loss █▅▁▁▂
wandb:          [fold3] epoch ▁▃▅▆█
wandb:           [fold3] loss █▄▄▂▂▁▆▂▃▂▂▂▂▂▂▁▁▂▁▂▂▁▂▂▂▇▁▁▂▄▅▃▂▂▁▂▁▁▃▁
wandb:             [fold3] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold3] score ▁▆▇██
wandb: [fold4] avg_train_loss █▂▁▁▁
wandb:   [fold4] avg_val_loss █▃▂▁▁
wandb:          [fold4] epoch ▁▃▅▆█
wandb:           [fold4] loss █▃▄▃▃▁▂▁▃▁▁▂▂▂▁▂▂▁▂▁▂▂▁▁▁▂▁▁▁▂▂▁▁▂▁▂▂▁▁▁
wandb:             [fold4] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold4] score ▁▅▆██
wandb: 
wandb: Run summary:
wandb: [fold0] avg_train_loss 0.00761
wandb:   [fold0] avg_val_loss 0.01254
wandb:          [fold0] epoch 5
wandb:           [fold0] loss 0.01159
wandb:             [fold0] lr 0.0
wandb:          [fold0] score 0.86382
wandb: [fold1] avg_train_loss 0.00835
wandb:   [fold1] avg_val_loss 0.01247
wandb:          [fold1] epoch 5
wandb:           [fold1] loss 0.01309
wandb:             [fold1] lr 0.0
wandb:          [fold1] score 0.85827
wandb: [fold2] avg_train_loss 0.00841
wandb:   [fold2] avg_val_loss 0.01238
wandb:          [fold2] epoch 5
wandb:           [fold2] loss 0.00277
wandb:             [fold2] lr 0.0
wandb:          [fold2] score 0.85881
wandb: [fold3] avg_train_loss 0.00813
wandb:   [fold3] avg_val_loss 0.01392
wandb:          [fold3] epoch 5
wandb:           [fold3] loss 0.00876
wandb:             [fold3] lr 0.0
wandb:          [fold3] score 0.85494
wandb: [fold4] avg_train_loss 0.00841
wandb:   [fold4] avg_val_loss 0.01209
wandb:          [fold4] epoch 5
wandb:           [fold4] loss 0.00718
wandb:             [fold4] lr 0.0
wandb:          [fold4] score 0.868
wandb: 
wandb: Synced microsoft/deberta-base: https://wandb.ai/feedback_prize_michael_and_wilson/NBME/runs/3qdrymmr
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20220319_065545-3qdrymmr/logs

# Run 2
========== CV ==========
Score: 0.8608
Waiting for W&B process to finish... (success).
 - 0.001 MB of 0.001 MB uploaded (0.000 MB deduped)
                                                                            

- Run history:
- [fold0] avg_train_loss █▂▂▁▁
- [fold0] avg_val_loss █▃▂▁▂
- [fold0] epoch ▁▃▅▆█
- [fold0] loss █▄▃▃▃▃▂▂▃▁▂▁▂▂▄▂▂▂▁▂▂▂▁▁▁▂▃▁▁▁▂▂▂▁▁▂▁▄▁▁
- [fold0] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold0] score ▁▆▆██
- [fold1] avg_train_loss █▂▁▁▁
- [fold1] avg_val_loss █▂▂▁▂
- [fold1] epoch ▁▃▅▆█
- [fold1] loss ▄▂▃█▆▃▇▃▂▃▄▃▃▄▁▂▂▂▂▂▁▁▂▂▂▂▃▂▂▁▂▃▄▆▃▂▁▂▁▂
- [fold1] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold1] score ▁▅▇▇█
- [fold2] avg_train_loss █▂▁▁▁
- [fold2] avg_val_loss █▃▁▁▁
- [fold2] epoch ▁▃▅▆█
- [fold2] loss ▆▆▃▆▄▃▅▇▃▃▃▆▂▂▂▂▂▃▂▂▁▁▁▃▁▂▃█▂▂▂▂▂▃▁▂▃▃▁▁
- [fold2] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold2] score ▁▆▇██
- [fold3] avg_train_loss █▂▁▁▁
- [fold3] avg_val_loss █▅▁▁▂
- [fold3] epoch ▁▃▅▆█
- [fold3] loss █▄▄▂▂▁▆▂▃▂▂▂▂▂▂▁▁▂▁▂▂▁▂▂▂▇▁▁▂▄▅▃▂▂▁▂▁▁▃▁
- [fold3] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold3] score ▁▆▇██
- [fold4] avg_train_loss █▂▁▁▁
- [fold4] avg_val_loss █▃▂▁▁
- [fold4] epoch ▁▃▅▆█
- [fold4] loss █▃▄▃▃▁▂▁▃▁▁▂▂▂▁▂▂▁▂▁▂▂▁▁▁▂▁▁▁▂▂▁▁▂▁▂▂▁▁▁
- [fold4] lr ███████▇▇▇▇▇▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold4] score ▁▅▆██
 
- Run summary:
- [fold0] avg_train_loss 0.00761
- [fold0] avg_val_loss 0.01254
- [fold0] epoch 5
- [fold0] loss 0.01159
- [fold0] lr 0.0
- [fold0] score 0.86382
- [fold1] avg_train_loss 0.00835
- [fold1] avg_val_loss 0.01247
- [fold1] epoch 5
- [fold1] loss 0.01309
- [fold1] lr 0.0
- [fold1] score 0.85827
- [fold2] avg_train_loss 0.00841
- [fold2] avg_val_loss 0.01238
- [fold2] epoch 5
- [fold2] loss 0.00277
- [fold2] lr 0.0
- [fold2] score 0.85881
- [fold3] avg_train_loss 0.00813
- [fold3] avg_val_loss 0.01392
- [fold3] epoch 5
- [fold3] loss 0.00876
- [fold3] lr 0.0
- [fold3] score 0.85494
- [fold4] avg_train_loss 0.00841
- [fold4] avg_val_loss 0.01209
- [fold4] epoch 5
- [fold4] loss 0.00718
- [fold4] lr 0.0
- [fold4] score 0.868

- Synced microsoft/deberta-base: https://wandb.ai/feedback_prize_michael_and_wilson/NBME/runs/1oa59l0m
- Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
- Find logs at: ./wandb/run-20220319_092035-1oa59l0m/logs

# run 3
========== CV ==========
Score: 0.8555
wandb: Waiting for W&B process to finish... (success).
wandb: - 0.001 MB of 0.001 MB uploaded (0.000 MB deduped)


### Run history:
- [fold0] avg_train_loss █▂▂▁▁
- [fold0] avg_val_loss █▃▂▁▂
- [fold0] epoch ▁▃▅▆█
- [fold0] loss █▇▄▃▅▃▂▁▃▄▂▃▁▂▁▂▁▂▁▄▄▃▃▃▁▂▂▁▁▁▂▂▁▂▂▂▁▂▁▅
- [fold0] lr ███████▇▇▇▇▇▇▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold0] score ▁▇▇██
- [fold1] avg_train_loss █▂▁▁▁
- [fold1] avg_val_loss █▃▁▁▂
- [fold1] epoch ▁▃▅▆█
- [fold1] loss █▃▃▄▄▂▃▂▁▁▂▁▅▃▂▂▁▂▂▁▂▁▂▂▂▂▃▂▂▂▁▃▁▂▃▄▁▂▁▁
- [fold1] lr ███████▇▇▇▇▇▇▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold1] score ▁▆▇██
- [fold2] avg_train_loss █▂▁▁▁
- [fold2] avg_val_loss █▃▁▁▁
- [fold2] epoch ▁▃▅▆█
- [fold2] loss █▅▂▄▃▂▂▃▂▂▂▂▂▂▂▂▂▂▂▃▂▁▂▁▁▄▁▃▂▂▂▂▁▂▁▂▁▂▁▁
- [fold2] lr ███████▇▇▇▇▇▇▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold2] score ▁▆▇██
- [fold3] avg_train_loss █▂▁▁▁
- [fold3] avg_val_loss █▃▂▁▂
- [fold3] epoch ▁▃▅▆█
- [fold3] loss █▄▃▂▂▁▁▂▂▂▃▃▁▂▁▁▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▂▁▁▁
- [fold3] lr ███████▇▇▇▇▇▇▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold3] score ▁▆▇██
- [fold4] avg_train_loss █▂▁▁▁
- [fold4] avg_val_loss █▂▂▂▁
- [fold4] epoch ▁▃▅▆█
- [fold4] loss █▃▇▄▃▂▄▄▂▁▃▂▂▂▂▃▂▁▃▂▂▂▂▁▃▂▂▁▁▂▁▂▁▁▂▃▁▂▂▂
- [fold4] lr ███████▇▇▇▇▇▇▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
- [fold4] score ▁▆▇▇█

### Run summary:
 [fold0] avg_train_loss 0.0079
- [fold0] avg_val_loss 0.01213
- [fold0] epoch 5
- [fold0] loss 0.01263
- [fold0] lr 0.0
- [fold0] score 0.85873
- [fold1] avg_train_loss 0.00895
- [fold1] avg_val_loss 0.01221
- [fold1] epoch 5
- [fold1] loss 0.01134
- [fold1] lr 0.0
- [fold1] score 0.85598
- [fold2] avg_train_loss 0.00914
- [fold2] avg_val_loss 0.01259
- [fold2] epoch 5
- [fold2] loss 0.00444
- [fold2] lr 0.0
- [fold2] score 0.85293
- [fold3] avg_train_loss 0.00877
- [fold3] avg_val_loss 0.01386
- [fold3] epoch 5
- [fold3] loss 0.00902
- [fold3] lr 0.0
- [fold3] score 0.85116
- [fold4] avg_train_loss 0.00874
- [fold4] avg_val_loss 0.01207
- [fold4] epoch 5
- [fold4] loss 0.00681
- [fold4] lr 0.0
- [fold4] score 0.85834
 
- Synced microsoft/deberta-base: https://wandb.ai/feedback_prize_michael_and_wilson/NBME/runs/3kx7fhdi
- Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
- Find logs at: ./wandb/run-20220319_114722-3kx7fhdi/logs

# run 4

========== CV ==========
Score: 0.8520

wandb: Waiting for W&B process to finish... (success).
wandb:                                                                                
wandb: 
wandb: Run history:
wandb: [fold0] avg_train_loss █▂▁▁▁
wandb:   [fold0] avg_val_loss █▃▁▁▁
wandb:          [fold0] epoch ▁▃▅▆█
wandb:           [fold0] loss █▄▄▄▅▂▂▁▂▂▂▂▂▂▁▁▂▁▂▂▄▂▁▂▂▂▁▁▃▂▁▃▁▂▁▂▂▂▁▁
wandb:             [fold0] lr ███████▇▇▇▇▇▆▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold0] score ▁▆███
wandb: [fold1] avg_train_loss █▂▁▁▁
wandb:   [fold1] avg_val_loss █▃▁▁▁
wandb:          [fold1] epoch ▁▃▅▆█
wandb:           [fold1] loss █▅▂▆▃▂▃▂▁▄▂▂▂▂▂▂▃▁▂▃▁▁▂▂▂▁▁▂▂▃▂▁▁▁▄▂▃▂▂▂
wandb:             [fold1] lr ███████▇▇▇▇▇▆▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold1] score ▁▆▇██
wandb: [fold2] avg_train_loss █▂▁▁▁
wandb:   [fold2] avg_val_loss █▄▂▁▁
wandb:          [fold2] epoch ▁▃▅▆█
wandb:           [fold2] loss █▃▄▂▂▂▂▂▂▂▂▂▁▁▁▂▁▁▁▂▁▁▁▁▂▁▁▂▁▁▂▁▁▁▂▁▁▂▁▁
wandb:             [fold2] lr ███████▇▇▇▇▇▆▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold2] score ▁▆▇██
wandb: [fold3] avg_train_loss █▂▁▁▁
wandb:   [fold3] avg_val_loss █▄▁▁▁
wandb:          [fold3] epoch ▁▃▅▆█
wandb:           [fold3] loss █▄▃▄▃▁▃▂▂▂▁▁▃▂▂▂▃▁▁▂▁▁▂▁▂▃▃▂▁▃▁▁▁▂▂▁▂▁▂▂
wandb:             [fold3] lr ███████▇▇▇▇▇▆▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold3] score ▁▆▇██
wandb: [fold4] avg_train_loss █▂▁▁▁
wandb:   [fold4] avg_val_loss █▃▃▁▁
wandb:          [fold4] epoch ▁▃▅▆█
wandb:           [fold4] loss █▅▃▄▃▄▂▃▂▁▁▃▂▁▂▂▁▁▂▂▂▂▁▁▂▂▁▃▁▁▁▂▁▃▁▁▂▂▂▁
wandb:             [fold4] lr ███████▇▇▇▇▇▆▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁
wandb:          [fold4] score ▁▇▇██
wandb: 
wandb: Run summary:
wandb: [fold0] avg_train_loss 0.00851
wandb:   [fold0] avg_val_loss 0.01194
wandb:          [fold0] epoch 5
wandb:           [fold0] loss 0.00883
wandb:             [fold0] lr 0.0
wandb:          [fold0] score 0.85726
wandb: [fold1] avg_train_loss 0.00987
wandb:   [fold1] avg_val_loss 0.01183
wandb:          [fold1] epoch 5
wandb:           [fold1] loss 0.00949
wandb:             [fold1] lr 0.0
wandb:          [fold1] score 0.85273
wandb: [fold2] avg_train_loss 0.0096
wandb:   [fold2] avg_val_loss 0.01281
wandb:          [fold2] epoch 5
wandb:           [fold2] loss 0.00331
wandb:             [fold2] lr 0.0
wandb:          [fold2] score 0.84688
wandb: [fold3] avg_train_loss 0.00924
wandb:   [fold3] avg_val_loss 0.01398
wandb:          [fold3] epoch 5
wandb:           [fold3] loss 0.00699
wandb:             [fold3] lr 0.0
wandb:          [fold3] score 0.84568
wandb: [fold4] avg_train_loss 0.00937
wandb:   [fold4] avg_val_loss 0.01191
wandb:          [fold4] epoch 5
wandb:           [fold4] loss 0.00748
wandb:             [fold4] lr 0.0
wandb:          [fold4] score 0.85722
wandb: 

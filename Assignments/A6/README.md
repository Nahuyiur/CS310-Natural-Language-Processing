## Introduction to scripts

## Submission requirements

You need to submit the following files:
- `model.py`
- `train.py`
- The resulted `bert_model.pt` file.


## Step 1. Run vocab.py

```bash
python vocab.py $DATA_FILE
```

`$DATA_FILE` is the path to the dataset file, e.g., `data_clean.txt`.

It will by default generate the vocabulary file `vocab.txt` and a binary file `input_ids_list.pkl`, which is a serialized Python list of all input ids.

## Step 2. Run batchify.py

```bash
python batchify.py
```

It will take the `input_ids_list.pkl` generated in Step 1 and produce two binary files `batches.pkl` and `batches_eval.pkl`. 

The former will be used for training, and the latter for evaluation (optional).

## Step 3. Implement model.py

Implement the `EncoderLayer` and `BERT` classes in `model.py`.

## Step 4. Implement and run train.py

Implement the training loop in `train.py`.

```bash
python train.py
```

You expect to see the training loss at the end of 1 epoch to be below 10.0, if everything implemented correctly.

Some testing output on 5 epochs:
```
Epoch 0/5, Batch 5490/5497, Loss 6.11, Remaining 0.33 sec for current epoch
Epoch 1/5, Batch 5490/5497, Loss 6.06, Remaining 0.33 sec for current epoch
Epoch 2/5, Batch 5490/5497, Loss 6.03, Remaining 0.33 sec for current epoch
Epoch 3/5, Batch 5490/5497, Loss 6.01, Remaining 0.33 sec for current epoch
Epoch 4/5, Batch 5490/5497, Loss 5.99, Remaining 0.33 sec for current epoch
```

It takes around 1200 seconds on a 8-core CPU, and should be 10x faster on a GPU.

The training script will take the `batches.pkl` file generated in Step 2 as input, and output the trained model to `bert_model.pt`.

## Step 5 (Optional). Run eval.py

```bash
python eval.py $MODEL_FILE
```

The script will evalute the model on `batches_eval.pkl` and output the accuracies: 

```
NSP Accuracy: 0.5000
Mask Accuracy: 0.1446
Total samples: 768
Total mask tokens: 3355
```

As you can see, the accuracies are not very good, which is expected as we are using a very small dataset and a very simple model.

No worries, the evaluation is just for testing purpose, and will not be used for grading.
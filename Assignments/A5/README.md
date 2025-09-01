## Introduction to scripts

- `parse_utils.py`: A superset of the `dep_utils.py` used for Lab 9. The extra content includes a `State` class and a `get_training_instances` function implemented, which are slightly different from those used in the lab but work similarly. 
- `get_vocab.py`: A script to generate vocabulary files from the CoNLL 2005 dataset.
- `get_train_data.py`: A script to generate training data to be saved to `.npy` files. There is a `FeatureExtractor` class that need be implemented.
- `train.py`: A script to train a dependency parser.
- `parser.py`: Containing a `Parser` class, whose `parse_sentence` method need be implemented.
- `model.py`: Containing a `BaseModel` and `WordPosModel` classes that need be implemented. 
- `evaluate.py`: A script to evaluate a trained model.

## Submission requirements
- Required Python files: `get_train_data.py`, `parser.py`, and `model.py`
  - `get_train_data.py` should be able to generate `.npy` data files properly.
  - `parser.py` should be able to parse the example sentence in `__main__` into the correct format (not necessarily the correct dependency tree).
  - `model.py` should contain at least `BaseModel` and `WordPosModel` properly implemented. The provided hyperparams (embedding dim, hidden dim, etc.) are just for reference, you can change them as you wish. 
- Required model files: All `.pt` files saved by `train.py`. Name them properly, e.g., `base_model.pt`, `wordpos_model.pt`.
- Optional other files: Any files you have created for the bonus tasks should come with good documentation.

## Step 1. Run get_vocab.py

```bash
python get_vocab.py $DATA_DIR 
```

It will by default generate three vocabulary files: `words_vocab.txt`, `pos_vocab.txt` and `deprel_vocab.txt`.


## Step 2. Implement and run get_train_data.py

After implementing the `FeatureExtractor` class, run the script:

```bash
python get_train_data.py $DATA_DIR/train.conll
```

It will by default generate two `.npy` data files: `input_train.npy` and `target_train.npy`.

## Step 3. Implement model.py and train.py

Implement the `BaseModel` and `WordPosModel` classes in `model.py`. 

Fill in the necessary model initialization and training code in `train.py`.

Then run the script:

```bash
python train.py --model $MODEL_NAME
```

It will start the training loop with 5 epochs by default, and save the model to a `.pt` after training is finished.

## Step 4. Implement parser.py

Implement the `Parser` class in `parser.py`, It has a `model` member that is an instance of `BaseModel` or `WordPosModel` (or other models for the bonus task), and a `parse_sentence` method that parses a sentence into a dependency tree.

The main body of `parser_sentence` method is a loop that iteratively performs transitions on the current state.
- Initialize the `state` with the input words.
- At each iteration, the `state` object is passed to the `get_input_repr_*` method of a `FeatureExtractor` object to get the input representation.
- Pass the input representation to the `model` object to get the probabilities of all possible next transition actions.
- Choose the next action with the highest probability (greedy decoding).
- Update the `state` by calling the corresponding method, `shift()`, `left_arc()`, `right_arc()` etc.

After implementing the `Parser` class, run the script:

```bash
python parser.py --model $MODEL_NAME
```

It will parse the example sentence in `__main__` and print the result.

## Step 5. Run evaluate.py

Run the script:

```bash
python evaluate.py --data $DATA_DIR --model $MODEL_NAME 
```

It will evaluate the trained model on the dev and test sets in `$DATA_DIR` and print the micro/macro level LAS (labeled) and UAS (unlabeled) scores.

**Note** that it is not required to have the `WordPOSModel` perform better than `BaseModel`, as it is not necessarily the case (hyperparams matter; overfitting happens; etc.). 


## About bonus tasks

For the bonus Task 6 (arc-eager approach), you need to modify the `State` class to change the behaviors of `left_arc()` and `right_arc()`, and add a new method `reduce()`. You also need to modify the `get_training_instances` function in `get_train_data.py`, so that it behaves in a "arg-eager" way. 

For the bonus Task 7 (Bi-LSTM model), you need to add a new class in `model.py`. You also need to make major changes to the `FeatureExtractor` class, because LSTM requires input in very different format.

Good luck!
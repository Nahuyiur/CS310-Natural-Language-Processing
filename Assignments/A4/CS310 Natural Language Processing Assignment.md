

# CS310 Natural Language Processing Assignment 

# 4: Long Short-Term Memory for Named Entity Recognition 

12310520 芮煜涵

## Basic requirement

We implemented the basic level 0 local classifier model as mentioned in the assignment file.

Here we adopt such parameters:

- `embedding_dim` = 100
- `hidden_dim` = 256

Here we train for 10 epochs, for that the epoch loss is quite small then, only 0.0005.

The F-1 scores during the first 5 training epochs on dev set:

![image-20250406211026979](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250406211026979.png)

 The final score on the test set, it's quite well, larger than 0.7.

![image-20250406211054673](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250406211054673.png)

## Bonus part

###  Implement the maximum entropy Markov model

Code in A4_MEMM.ipynb.

We create an embedding layer for all the NER tags, and use the hidden states of previous tags for predicting the next one.

We keep the parameters the same as the basic level 0 model.

Here are the training loss and the evaluation on test set:

![image-20250406211508378](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250406211508378.png)

The training time is much shorter than the level 0 model, but the performance is a little bit worse(on test set F1 score=0.6391).

### Implement beam search for decoding at testing time

Code in A4_beam_search.ipynb.

We load the model trained before, only change the method for decoding and then evaluate this strategy.

![image-20250406211727552](/Users/ruiyuhan/Library/Application Support/typora-user-images/image-20250406211727552.png)

But even we increase the beam_with from 5 to 100, there is no change in performance.

Here are the possible reasons:

The lack of performance difference between beam search and greedy search in this context likely stems from the simplicity of the NER task or the robustness of the bidirectional LSTM model, where local optimal choices (greedy) align closely with global optima (beam search). This suggests that the model's predictions are already strong, and additional exploration via beam search does not significantly improve the F-1 score. 
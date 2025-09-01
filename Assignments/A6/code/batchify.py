from typing import List, Dict
from tqdm import tqdm
import random
import pickle
import os

from config import *
"""
Note that word_to_id and id_to_word are loaded from config.py
"""

def make_batch(input_ids_list: List[int], batch_size: int, word_to_id: Dict):
    batch = []
    positive = negative = 0
    
    while positive != batch_size/2 or negative != batch_size/2:
        sent_a_index, sent_b_index= random.randrange(len(input_ids_list)), random.randrange(len(input_ids_list))
        tokens_a, tokens_b= input_ids_list[sent_a_index], input_ids_list[sent_b_index]

        input_ids = [word_to_id['[CLS]']] + tokens_a + [word_to_id['[SEP]']] + tokens_b + [word_to_id['[SEP]']]
        segment_ids = [1] * (1 + len(tokens_a) + 1) + [2] * (len(tokens_b) + 1)

        # The following code is used for the Masked Language Modeling (MLM) task.
        n_pred =  min(MAX_PRED, max(1, int(round(len(input_ids) * 0.15)))) # Predict at most 15 % of tokens in one sentence
        masked_candidates_pos = [i for i, token in enumerate(input_ids)
                          if token != word_to_id['[CLS]'] and token != word_to_id['[SEP]']]
        random.shuffle(masked_candidates_pos)
        masked_tokens, masked_pos = [], []
        for pos in masked_candidates_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])

            # Throw a dice to decide if you want to replace the token with [MASK], random word, or remain the same
            if random.random() < 0.8:  # 80% replaced with [MASK]
                input_ids[pos] = word_to_id['[MASK]'] # make mask
            elif random.random() < 0.5:  # (1 - 80%) * 0.5 = 10% replaced with random word
                index = random.randint(4, VOCAB_SIZE - 1) # random index in vocabulary, except for the special tokens
                input_ids[pos] = word_to_id[id_to_word[index]] # replace

        # Zero padding (100% - 15%) of thetokens
        if MAX_PRED > n_pred:
            n_pad = MAX_PRED - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # The following code is used for the Next Sentence Prediction (NSP) task.
        if sent_a_index + 1 == sent_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # Is Next
            positive += 1
        elif sent_a_index + 1 != sent_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # Is Not Next
            negative += 1
    
    # Get max length in current batch, and make zero paddings
    max_len = max([len(x[0]) for x in batch])
    for b in batch:
        b[0] += [0] * (max_len - len(b[0])) # extend input_ids
        b[1] += [0] * (max_len - len(b[1])) # extend segment_ids

    return batch


if __name__ == '__main__':
    assert os.path.exists('input_ids_list.pkl')
    with open('input_ids_list.pkl', 'rb') as f:
        input_ids_list = pickle.load(f)
    
    n_batches = len(input_ids_list) // BATCH_SIZE
    batches = []
    for i in tqdm(range(n_batches)):
        batch = make_batch(input_ids_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE], BATCH_SIZE, word_to_id)
        # toss if batch is too long
        if len(batch[0][0]) > MAX_LEN:
            continue
        batches.append(batch)
    
    with open('batches.pkl', 'wb') as f:
        pickle.dump(batches, f)
    
    # Sample some batches for validation
    random.shuffle(batches)
    with open('batches_eval.pkl', 'wb') as f:
        pickle.dump(batches[:128], f)
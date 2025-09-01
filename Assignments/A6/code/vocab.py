from typing import List
from tqdm import tqdm
from collections import Counter
import sys
import os
import pickle

def tokenizer(text: str) -> List[str]:
    '''
    A simple Chinese tokenizer that splits a sentence into a list of words (characters, 汉字).
    '''
    return list(text)

def build_vocab(train_file, tokenizer):
    '''
    Build a vocabulary from a training file.
    '''
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    counter = Counter()    
    word_to_id = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
    print('building vocabulary...')
    for line in tqdm(lines):
        words = tokenizer(line.strip())
        for word in words:
            counter[word] += 1
            if counter[word] >= 3:
                if word not in word_to_id:
                    word_to_id[word] = len(word_to_id)

    print('saving text data to list of token ids...')
    input_ids_list = []
    for line in tqdm(lines):
        words = tokenizer(line.strip())
        input_ids = []
        for w in words:
            if w in word_to_id:
                input_ids.append(word_to_id[w])
            else:
                input_ids.append(word_to_id['[UNK]'])
        input_ids_list.append(input_ids)

    return word_to_id, input_ids_list


def save_vocab(word_to_id, save_path):
    '''
    Save the vocabulary to a file.
    '''
    with open(save_path, 'w', encoding='utf-8') as f:
        for word, id in word_to_id.items():
            f.write(f'{word}\t{id}\n')

def load_vocab(vocab_path):
    '''
    Load the vocabulary from a file.
    '''
    word_to_id = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, id = line.strip().split('\t')
            word_to_id[word] = int(id)
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


if __name__ == '__main__':
    train_file = sys.argv[1]
    vocab_path = 'vocab.txt'

    word_to_id, tokens_list = build_vocab(train_file, tokenizer)
    save_vocab(word_to_id, vocab_path)

    # Save tokens_list to a pickle file
    pickle.dump(tokens_list, open('input_ids_list.pkl', 'wb'))
# from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import os
from tqdm import tqdm
from typing import Tuple, List
import ast
import numpy as np
import argparse

if os.path.exists('parse_utils.py'):
    from parse_utils import conll_reader, get_training_instances
else:
    raise Exception('Could not find parse_utils.py or dep_utils.py')

argparser = argparse.ArgumentParser()
argparser.add_argument('train_data', help='Path to the training data')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')
argparser.add_argument('--output_data', default='input_train.npy')
argparser.add_argument('--output_target', default='target_train.npy')


class FeatureExtractor(object):
    def __init__(self, word_vocab_file, pos_vocab_file, rel_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)
        self.pos_vocab = self.read_vocab(pos_vocab_file)
        self.rel_vocab = self.create_rel_vocab(rel_vocab_file)

    def create_rel_vocab(self, rel_vocab_file):
        vocab = {}
        vocab[('shift', None)] = 0
        for line in rel_vocab_file:
            key_s, index_s = line.strip().split('\t')
            index = int(index_s)
            key = ast.literal_eval(key_s) # e.g., "(\'left_arc\', \'csubj\')" -> ('left_arc', 'csubj')
            vocab[key] = index + 1 # the original rel vocab file starts from 0
        return vocab

    def read_vocab(self, vocab_file):
        vocab = {}
        for line in vocab_file:
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab

    def get_input_repr_word(self, words, pos, state):
        """
        words: list of words in a dependency tree
        pos: list of pos tags in a dependency tree
        state: a State object, which is obtained from get_training_instances()
        Return: a numpy array of size 6, in which the first 3 elements are the IDs of the top 3 words on the stack, and the last 3 elements are the IDs of the top 3 words on the buffer
        """
        rep = np.zeros(6)

        for i in range(3):
            ### START YOUR CODE ###
            # TODO: Go through the top 3 words on stack
            # If stack is empty, use <NULL> as the word; if stack contains <3 words, also fill with <NULL>
            # If word is not in word_vocab, use <UNK> as the word
            if len(state.stack)>i:
                word=words[state.stack[-(i+1)]]
            else:
                word="<NULL>"
            if word not in self.word_vocab:
                word="<UNK>"
            rep[i] = self.word_vocab[word]

            # TODO: Go through the top 3 words on buffer, similar to the above
            if len(state.buffer) > i:
                word = words[state.buffer[-(i+1)]]
            else:
                word = "<NULL>"
            if word not in self.word_vocab:
                word = "<UNK>"
            rep[i+3] = self.word_vocab[word]
            ### END YOUR CODE ###

        return rep
    
    def get_input_repr_wordpos(self, words, pos, state):
        """
        Return: a numpy array of size 12, in which the first 6 elements are the words IDs of the top 3 words on the stack plus the top 3 on the buffer; the last 6 elements are the POS IDs of the top 3 words on the stack plus the top 3 on the buffer
        """
        rep = np.zeros(12)

        for i in range(3):
            ### START YOUR CODE ###
            # 栈
            if len(state.stack) > i:
                word = words[state.stack[-(i+1)]]
                tag = pos[state.stack[-(i+1)]]
            else:
                word = "<NULL>"
                tag = "<NULL>"
            if word not in self.word_vocab:
                word = "<UNK>"
            if tag not in self.pos_vocab:
                tag = "<UNK>"
            rep[i] = self.word_vocab[word]
            rep[i+6] = self.pos_vocab[tag]

            # 缓冲区
            if len(state.buffer) > i:
                word = words[state.buffer[-(i+1)]]
                tag = pos[state.buffer[-(i+1)]]
            else:
                word = "<NULL>"
                tag = "<NULL>"
            if word not in self.word_vocab:
                word = "<UNK>"
            if tag not in self.pos_vocab:
                tag = "<UNK>"
            rep[i+3] = self.word_vocab[word]
            rep[i+9] = self.pos_vocab[tag]
            ### END YOUR CODE ###

        return rep

    def get_target_repr(self, action):
        # action is a tuple of (transition, label)
        # Get its index from self.rel_vocab
        return np.array(self.rel_vocab[action])


def get_training_matrices(extractor, input_filename: str, n=np.inf) -> Tuple[List, List]:
    inputs = []
    targets = []
    count = 0
    with open(input_filename, "r") as in_file:
        dtrees = list(conll_reader(in_file))
    for dtree in tqdm(dtrees, total=min(len(dtrees), n)):
        words = dtree.words()
        pos = dtree.pos()
        for state, action in get_training_instances(dtree):

            ### START YOUR CODE ###
            # TODO: Call extractor.get_input_repr_*() and append the result to inputs
            inputs.append(extractor.get_input_repr_wordpos(words, pos, state))
            ### END YOUR CODE ###

            targets.append(extractor.get_target_repr(action))
        count += 1
        if count >= n:
            break
    return inputs, targets 


if __name__ == "__main__":
    args = argparser.parse_args()
    input_file = args.train_data
    assert os.path.exists(input_file)

    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}')
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    print("Starting feature extraction...")

    inputs, targets = get_training_matrices(extractor, input_file)
    inputs = np.stack(inputs)
    targets = np.stack(targets)
    np.save(args.output_data, inputs)
    np.save(args.output_target, targets)

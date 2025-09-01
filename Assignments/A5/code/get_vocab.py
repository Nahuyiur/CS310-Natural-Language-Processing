import sys
import os
import argparse
from collections import defaultdict
from typing import List

if os.path.exists('parse_utils.py'):
    from parse_utils import conll_reader, DependencyTree
elif os.path.exists('dep_utils.py'):
    from dep_utils import conll_reader, DependencyTree
else:
    raise Exception('Could not find parse_utils.py or dep_utils.py')


argparser = argparse.ArgumentParser()
argparser.add_argument('data_dir', type=str, help='Directory containing the CoNLL data')
argparser.add_argument('--words_vocab', default='words_vocab.txt', help='File to write word indices to')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt', help='File to write POS indices to')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt', help='File to write relation indices to')


def get_vocab(dep_trees: List[DependencyTree]):
    words_set = defaultdict(int)
    pos_set = set()
    rel_set = set()
    for dtree in dep_trees:
        for ident, node in dtree.deprels.items():
            if node.pos != "CD" and node.pos != "NNP": # remove numbers (e.g., ) and proper nouns
                words_set[node.word.lower()] += 1
            pos_set.add(node.pos)
            # add relation 
            if node.deprel == 'root':
                rel_set.add(('right_arc', 'root'))
            else:
                rel_set.add(('left_arc', node.deprel))
                rel_set.add(('right_arc', node.deprel))

    words_set = set(x for x in words_set if words_set[x] > 1)
    words_list = ["<CD>", "<NNP>", "<UNK>", "<ROOT>", "<NULL>"] + list(words_set)
    pos_list = ["<UNK>", "<ROOT>", "<NULL>"] + list(pos_set)
    rel_list = list(rel_set)

    return words_list, pos_list, rel_list


if __name__ == "__main__":
    args = argparser.parse_args()
    if not os.path.exists(args.data_dir):
        raise Exception(f'Data directory {args.data_dir} does not exist')

    dep_trees = []
    train_filename = 'train.conll'
    input_path = os.path.join(args.data_dir, train_filename)
    with open(input_path, 'r') as input_file:
        dep_trees.extend(conll_reader(input_file))
    print(f"Read {len(dep_trees)} trees in {train_filename}")

    words_list, pos_list, rel_list = get_vocab(dep_trees)
    print("Writing word indices...")
    with open(args.words_vocab, 'w') as f:
        for index, word in enumerate(words_list):
            f.write("{}\t{}\n".format(word, index))

    print("Writing POS indices...")
    with open(args.pos_vocab, 'w') as f:
        for index, pos in enumerate(pos_list):
            f.write("{}\t{}\n".format(pos, index))
    
    print('Wrting dependency relations...')
    with open(args.rel_vocab, 'w') as f:
        for index, rel in enumerate(rel_list):
            f.write(f'{rel}\t{index}\n')

import time
import sys
import os
from tqdm import tqdm
import argparse

from parser import Parser
from get_train_data import FeatureExtractor
from parse_utils import conll_reader


argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='model.pt')
argparser.add_argument('--data', type=str, default='data', help='path to data directory')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')


def compare_parser(target, predict):
    target_unlabeled = set((d.id, d.head) for d in target.deprels.values())
    target_labeled = set((d.id, d.head, d.deprel) for d in target.deprels.values())
    predict_unlabeled = set((d.id, d.head) for d in predict.deprels.values())
    predict_labeled = set((d.id, d.head, d.deprel) for d in predict.deprels.values())

    labeled_correct = len(predict_labeled.intersection(target_labeled))
    unlabeled_correct = len(predict_unlabeled.intersection(target_unlabeled))
    num_words = len(predict_labeled)
    if num_words == 0:
        print("Warning: sentence with no words")
    return labeled_correct, unlabeled_correct, num_words


if __name__ == "__main__":
    # parse arguments
    args = argparser.parse_args()
    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}')
        sys.exit(1)

    start = time.time()
    print("Start time: ", start)
    # convert start time to readable format
    print("Start time: ", time.ctime(start))

    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    parser = Parser(extractor, args.model)

    total_labeled_correct = 0
    total_unlabeled_correct = 0
    total_words = 0
    las_list = []
    uas_list = []

    if not os.path.exists(args.data):
        print(f'Error: directory {args.data} not found')
        sys.exit(1)
    EVAL_FILES = [os.path.join(args.data, 'dev.conll'), 
                  os.path.join(args.data, 'test.conll')]

    for eval_file in EVAL_FILES:
        print(f'Evaluating on {eval_file}')
        with open(eval_file, 'r') as in_file:
            dtrees = list(conll_reader(in_file))
            for dtree in tqdm(dtrees):
                words = dtree.words()
                pos = dtree.pos()
                predict = parser.parse_sentence(words, pos)

                labeled_correct, unlabeled_correct, num_words = compare_parser(dtree, predict)
                las_s = labeled_correct / float(num_words)
                uas_s = unlabeled_correct / float(num_words)
                las_list.append(las_s)
                uas_list.append(uas_s)

                total_labeled_correct += labeled_correct
                total_unlabeled_correct += unlabeled_correct
                total_words += num_words

        las_micro = total_labeled_correct / float(total_words)
        uas_micro = total_unlabeled_correct / float(total_words)
        las_macro = sum(las_list) / len(las_list)
        uas_macro = sum(uas_list) / len(uas_list)

        print("{} sentence.".format(len(las_list)))
        print("Micro Avg. Labeled Attachment Score: {}".format(las_micro))
        print("Micro Avg. Unlabeled Attachment Score: {}".format(uas_micro))
        print("Macro Avg. Labeled Attachment Score: {}".format(las_macro))
        print("Macro Avg. Unlabeled Attachment Score: {}".format(uas_macro))
        print()

    # time
    print("Time: {}".format(time.time() - start))

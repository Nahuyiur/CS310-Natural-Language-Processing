import sys
import numpy as np
import torch
import argparse

from model import BaseModel, WordPOSModel
from parse_utils import DependencyArc, DependencyTree, State, parse_conll_relation
from get_train_data import FeatureExtractor

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='model.pt')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')


class Parser(object):
    def __init__(self, extractor: FeatureExtractor, model_file: str):
        ### START YOUR CODE ###
        # TODO: Initialize the model
        word_vocab_size = len(extractor.word_vocab)
        pos_vocab_size = len(extractor.pos_vocab)
        output_size = len(extractor.rel_vocab)

        self.model = None
        if "base" in model_file.lower():
            print("Loading BaseModel...")
            self.model = BaseModel(word_vocab_size, output_size)
        else:
            print("Loading WordPOSModel...")
            self.model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
        ### END YOUR CODE ###

        self.model.load_state_dict(torch.load(model_file, weights_only=True))
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict(
            [(index, action) for (action, index) in extractor.rel_vocab.items()]
        ) # 将 动作的索引 映射到对应的 动作（例如 "shift", "left_arc", "right_arc" 等）

    def get_legal_transitions(self, state):
        legal_transitions=[]
        if len(state.buffer) > 1:
            legal_transitions.append(("shift", None))

        if len(state.stack) >= 1:
            for rel in self.extractor.rel_vocab:
                if rel[0] == "right_arc":
                    legal_transitions.append(rel)

        if len(state.stack) >= 2:
            for rel in self.extractor.rel_vocab:
                if rel[0] == "left_arc":
                    legal_transitions.append(rel)

        return legal_transitions
    
    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))  # 初始化解析状态，句子从1开始，0是ROOT
        state.stack.append(0)  # 将 <ROOT> 节点加入栈中，栈起始状态为 [0]，表示根节点

        while state.buffer and state.stack:
            # 提取当前状态的特征
            if isinstance(self.model, BaseModel):
                current_state = self.extractor.get_input_repr_word(words, pos, state)
            else:
                current_state = self.extractor.get_input_repr_wordpos(words, pos, state)

            with torch.no_grad():
                input_tensor = torch.from_numpy(current_state).unsqueeze(0).long() # 转为 tensor 格式并添加 batch 维度
                word_indices = input_tensor[:, :6]  # 取前6列
                pos_indices = input_tensor[:, 6:]   # 取后6列
                prediction = self.model((word_indices,pos_indices))  # 获取模型预测结果
            
            # 从合法的动作中选择得分最高的动作
            legal_predictions = self.get_legal_transitions(state)

            best_score = float('-inf')
            best_action = None
            for action in legal_predictions:
                action_idx = self.extractor.rel_vocab[action]
                score = prediction[0, action_idx].item()
                if score > best_score:
                    best_score = score
                    best_action = action
            
            if best_action[0] == "shift":
                state.shift()
            elif best_action[0] == "left_arc":
                state.left_arc(best_action[1])
            elif best_action[0] == "right_arc":
                state.right_arc(best_action[1])

        # TODO: Go through each relation in state.deps and add it to the tree by calling tree.add_deprel()
        tree = DependencyTree()
        for head, modifier, relation in state.deps:
            # 创建 DependencyArc 对象
            word = words[modifier] if modifier < len(words) else None
            pos_tag = pos[modifier] if modifier < len(pos) else None
            dep_arc = DependencyArc(
                word_id=modifier,
                word=word,
                pos=pos_tag,
                head=head,
                deprel=relation)
            tree.add_deprel(dep_arc)
        
        return tree


if __name__ == "__main__":
    args = argparser.parse_args()
    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}')
        sys.exit(1)
    
    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    parser = Parser(extractor, args.model)

    # Test an example sentence, 3rd example from dev.conll
    words = [None, 'The', 'bill', 'intends', 'to', 'restrict', 'the', 'RTC', 'to', 'Treasury', 'borrowings', 'only', ',', 'unless', 'the', 'agency', 'receives', 'specific', 'congressional', 'authorization', '.']
    pos = [None, 'DT', 'NN', 'VBZ', 'TO', 'VB', 'DT', 'NNP', 'TO', 'NNP', 'NNS', 'RB', ',', 'IN', 'DT', 'NN', 'VBZ', 'JJ', 'JJ', 'NN', '.']

    tree = parser.parse_sentence(words, pos)
    print(tree)
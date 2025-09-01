import sys
import copy
from collections import defaultdict
from typing import List, Tuple


class DependencyArc(object):
    """
    Represent a single dependency arc:
    """
    def __init__(self, word_id, word, pos, head, deprel):
        self.id = word_id
        self.word = word
        self.pos = pos
        self.head = head
        self.deprel = deprel
    
    def __str__(self) -> str:
        return "{d.id}\t{d.word}\t_\t_\t{d.pos}\t_\t{d.head}\t{d.deprel}\t_\t_".format(d=self)


def parse_conll_relation(s):
    fields = s.split("\t")
    word_id_str, word, lemma, upos, pos, feats, head_str, deprel, deps, misc = fields
    word_id = int(word_id_str)
    head = int(head_str)
    return DependencyArc(word_id, word, pos, head, deprel)


class DependencyTree(object):
    def __init__(self):
        self.deprels = {}
        self.root = None
        self.parent_to_children = defaultdict(list)

    def add_deprel(self, deprel):
        self.deprels[deprel.id] = deprel
        self.parent_to_children[deprel.head].append(deprel.id)
        if deprel.head == 0:
            self.root = deprel.id

    def __str__(self):
        deprels = [v for (k, v) in sorted(self.deprels.items())]
        return "\n".join(str(deprel) for deprel in deprels)
    
    def print_tree(self, parent=None):
        if not parent:
            return self.print_tree(parent=self.root)

        if self.deprels[parent].head == parent:
            return self.deprels[parent].word

        children = [self.print_tree(child) for child in self.parent_to_children[parent]]
        child_str = " ".join(children)
        return "({} {})".format(self.deprels[parent].word, child_str)

    def words(self):
        return [None] + [x.word for (i, x) in self.deprels.items()]

    def pos(self):
        return [None] + [x.pos for (i, x) in self.deprels.items()]
    
    def from_string(s):
        dtree = DependencyTree()
        for line in s.split("\n"):
            if line:
                dtree.add_deprel(parse_conll_relation(line))
        return dtree


def conll_reader(input_file):
    current_deps = DependencyTree()
    while True:
        line = input_file.readline().strip()
        if not line and current_deps:
            yield current_deps
            current_deps = DependencyTree()
            line = input_file.readline().strip()
            if not line:
                break
        current_deps.add_deprel(parse_conll_relation(line))


class State(object):
    def __init__(self, sentence=[]):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(sentence) # Arc-eager的buffer是正序
        self.deps = set()

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        # buffer[0] --> stack[-1]
        head = self.buffer[0]
        dependent = self.stack.pop()
        self.deps.add((head, dependent, label))

    def right_arc(self, label):
        # stack[-1] --> buffer[0]，并shift buffer[0]到stack
        head = self.stack[-1]
        dependent = self.buffer.pop(0)
        self.deps.add((head, dependent, label))
        self.stack.append(dependent)
    
    def reduce(self):
        # 仅pop stack[-1]
        self.stack.pop()

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)


class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None
    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_tree: DependencyTree) -> List[Tuple[State, Tuple[str, str]]]:
    deprels = dep_tree.deprels
    sorted_nodes = [k for k, v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)  # 加入ROOT

    childcount = defaultdict(int)
    for ident, node in deprels.items():
        childcount[node.head] += 1

    seq = []
    while state.buffer or len(state.stack) > 1:
        if not state.stack:
            # stack空了，只能shift
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
            continue

        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]
        
        if state.buffer:
            bufferword = deprels[state.buffer[0]]
        else:
            bufferword = None

        # 根据Arc-Eager标准，优先判断动作
        if bufferword and bufferword.head == stackword.id:
            # stack[-1] --> buffer[0]，Right-Arc
            seq.append((copy.deepcopy(state), ("right_arc", bufferword.deprel)))
            state.right_arc(bufferword.deprel)
            childcount[stackword.id] -= 1
        elif bufferword and stackword.head == bufferword.id:
            # buffer[0] --> stack[-1]，Left-Arc
            seq.append((copy.deepcopy(state), ("left_arc", stackword.deprel)))
            state.left_arc(stackword.deprel)
            childcount[bufferword.id] -= 1
        elif childcount[stackword.id] == 0:
            # 如果stack[-1]所有子节点已经处理完了，Reduce
            seq.append((copy.deepcopy(state), ("reduce", None)))
            state.reduce()
        else:
            # 否则Shift
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()

    return seq

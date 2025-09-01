import sys
from collections import defaultdict


class DependencyArc(object):
    """
    Represent a single dependency arc:
    """
    def __init__(self, word_id, word, pos, head, deprel):
        self.id = word_id # 词在句子中的位置
        self.word = word # 词形
        self.pos = pos # 词性
        self.head = head # 词的父节点
        self.deprel = deprel # 依存关系
    
    def __str__(self) -> str:
        return "{d.id}\t{d.word}\t_\t_\t{d.pos}\t_\t{d.head}\t{d.deprel}\t_\t_".format(d=self)


def parse_conll_relation(s):
    fields = s.split("\t") # 解析CONLL格式的文本，然后只选取了关心的行
    word_id_str, word, lemma, upos, pos, feats, head_str, deprel, deps, misc = fields
    word_id = int(word_id_str)
    head = int(head_str)
    return DependencyArc(word_id, word, pos, head, deprel)


class DependencyTree(object):
    def __init__(self):
        self.deprels = {} # 一个字典，将句子中每个词（token）的 ID 映射到它对应的 DependencyArc 对象
        self.root = None # 根节点的 head 字段为 0
        self.parent_to_children = defaultdict(list) # 一个 defaultdict(list)，用来存储父节点到子节点的映射

    def add_deprel(self, deprel): # 把一个新的依存关系添加到句子中
        self.deprels[deprel.id] = deprel # 把这个词的id映射到它的依存关系对象
        self.parent_to_children[deprel.head].append(deprel.id) # 把当前词的 ID 加入到它父节点（deprel.head）对应的子列表里。
        if deprel.head == 0:
            self.root = deprel.id

    def __str__(self): # 定义了当你 print(dtree) 时的输出格式
        deprels = [v for (k, v) in sorted(self.deprels.items())] # 先按 token ID 排序 (sorted(self.deprels.items()))，保证输出的行顺序与原句子顺序一致
        return "\n".join(str(deprel) for deprel in deprels) # 把依存关系转换成字符串，用换行符连接
    
    def print_tree(self, parent=None):
        if not parent:
            return self.print_tree(parent=self.root)

        if self.deprels[parent].head == parent: # 如果某个节点的 head 等于它自己（在某些语料中根节点或特定结构可能如此），就直接返回该词的表面形式。
            return self.deprels[parent].word

        children = [self.print_tree(child) for child in self.parent_to_children[parent]] # 对当前parent的子节点都递归调用
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
        if not line and current_deps: # 读到空行，就返回当前的依存树；并准备新的空依存树
            yield current_deps
            current_deps = DependencyTree()
            line = input_file.readline().strip()
            if not line:
                break
        current_deps.add_deprel(parse_conll_relation(line)) 


if __name__ == "__main__":
    with open(sys.argv[1], "r") as in_file:
        relations = set()
        for deps in conll_reader(in_file):
            for deprel in deps.deprels.values():
                relations.add(deprel.deprel)
            print(deps.print_tree())

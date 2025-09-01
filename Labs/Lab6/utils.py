import numpy as np
from tqdm import tqdm


DOCSTART_LITERAL = '-DOCSTART-'


def read_ner_data_from_connl(path_to_file):
    words = []
    tags = []

    with open(path_to_file, 'r', encoding='utf-8') as file:
        for line in file:
            splitted = line.split()
            if len(splitted) == 0:
                continue
            word = splitted[0]
            if word == DOCSTART_LITERAL:
                continue
            entity = splitted[-1]
            words.append(word)
            tags.append(entity)
        return words, tags


def get_batch(words, labels, size):
    for i in range(0, len(labels), size):
        yield (words[i:i + size], labels[i:i + size])


def load_embedding_dict(vec_path):
    embeddings_index = dict()
    with open(vec_path, 'r', encoding='UTF-8') as file:
        for line in tqdm(file.readlines()):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index


def get_tag_indices_from_scores(scores: np.ndarray):
    predicted = []
    for i in range(scores.shape[0]):
        predicted.append(int(np.argmax(scores[i])))
    return predicted


class Indexer:
    def __init__(self, elements):
        self._element_to_index = {"<UNKNOWN>": 0}
        for x in elements:
            if x not in self._element_to_index:
                self._element_to_index[x] = len(self._element_to_index)
        self._index_to_element = {v: k for k,v in self._element_to_index.items()}

    def get_element_to_index_dict(self):
        return self._element_to_index

    def element_to_index(self, element):
        return self._element_to_index.get(element, 0)

    def index_to_element(self, index):
        return self._index_to_element[index]

    def elements_to_indices(self, elements):
        return [self.element_to_index(x) for x in elements]

    def indices_to_elements(self, indexes):
        return [self.index_to_element(x) for x in indexes]

    def __len__(self):
        return len(self._element_to_index)

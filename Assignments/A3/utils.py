import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


DOCSTART_LITERAL = '-DOCSTART-'


def read_ner_data_from_connl(path_to_file):
    # 从CoNLL格式的文件中读取命名实体识别（NER）数据
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


def get_batched(words, labels, size):
    # 将单词和标签列表按指定大小分批返回
    for i in range(0, len(labels), size):
        yield (words[i:i + size], labels[i:i + size])


def load_embedding_dict(vec_path):
    # 返回词嵌入字典
    embeddings_index = dict()
    with open(vec_path, 'r', encoding='UTF-8') as file:
        for line in tqdm(file.readlines()):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index


def get_tag_indices_from_scores(scores: np.ndarray):
    # 从模型输出的得分矩阵中提取预测的标签索引
    predicted = []
    for i in range(scores.shape[0]):
        predicted.append(int(np.argmax(scores[i])))
    return predicted


def build_training_visualization(model_name, train_metrics, losses, validation_metrics, path_to_save=None):
    # 绘制训练过程中的可视化图表
    figure = plt.figure(figsize=(20, 30))
    figure.suptitle(f'Visualizations of {model_name} training progress', fontsize=16)

    ax1 = figure.add_subplot(3, 1, 1)
    ax1.plot(losses)
    ax1.set_title("Loss through epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2 = figure.add_subplot(3, 1, 2)
    for metric, results in validation_metrics.items():
        ax2.plot(results, label=metric)
    ax2.legend(loc='upper left')
    ax2.set_title("Metrics through epochs")
    ax2.set_xlabel("Epochs")

    ax3 = figure.add_subplot(3, 1, 3)
    for metric, results in train_metrics.items():
        ax3.plot(results, label=metric)
    ax3.legend(loc='upper left')
    ax3.set_title("Results on dev set through epochs")
    ax3.set_xlabel("Epochs")

    if path_to_save:
        figure.savefig(path_to_save)
import torch.nn as nn
import torch

class BaseModel(nn.Module):
    def __init__(self, word_vocab_size, output_size):
        super(BaseModel, self).__init__()
        ### START YOUR CODE ###
        word_dim=100
        hidden_dim=200

        self.word_embed = nn.Embedding(word_vocab_size, word_dim)
        self.hidden=nn.Linear(6*word_dim, hidden_dim)
        self.dropout=nn.Dropout(0.3)
        self.output=nn.Linear(hidden_dim, output_size)
        ### END YOUR CODE ###
    
    def forward(self, x):
        ### START YOUR CODE ###
        # x: [batch_size, 12] 取前6列
        if isinstance(x, tuple):
            # 如果是 (word_ids, pos_ids) 的tuple，取 word_ids
            word_ids, _ = x
        else:
            # 否则，直接就是 word_ids，取前6列
            word_ids = x[:, :6]

        embeds = self.word_embed(word_ids) # (batch_size, 6, 100)
        x = embeds.view(embeds.size(0), -1) # (batch_size, 600)

        x = torch.relu(self.hidden(x)) # (batch_size, 200)
        x = self.dropout(x) # (batch_size, 200)
        x = self.output(x) # (batch_size, output_size)
        ### END YOUR CODE ###
        return torch.log_softmax(x, dim=1)


class WordPOSModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        super(WordPOSModel, self).__init__()
        ### START YOUR CODE ###
        word_dim=100
        hidden_dim=200
        pos_dim=25 #每个 POS 标签（词性）对应的嵌入向量的维度（embedding size）

        self.word_embed = nn.Embedding(word_vocab_size, word_dim)
        self.pos_embed = nn.Embedding(pos_vocab_size, pos_dim)

        self.hidden = nn.Linear(6 * (word_dim + pos_dim), hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_dim, output_size)
        ### END YOUR CODE ###

    def forward(self, x):
        ### START YOUR CODE ###
        # x: (batch_size, 12) = 6 word IDs + 6 pos IDs

        # 分开词和pos的部分
        word_ids,pos_ids=x

        word_embeds = self.word_embed(word_ids)  # (batch_size, 6, 100)
        pos_embeds = self.pos_embed(pos_ids)     # (batch_size, 6, 25)

        combined = torch.cat([word_embeds, pos_embeds], dim=-1)  # (batch_size, 6, 125)
        x = combined.view(combined.size(0), -1)  # (batch_size, 750)

        x = torch.relu(self.hidden(x))  # (batch_size, 200)
        x = self.dropout(x)
        x = self.output(x)  # (batch_size, output_size)
        ### END YOUR CODE ###
        return torch.log_softmax(x, dim=1)  # 返回 log probabilities

from vocab import load_vocab

# Hyperparameters
MAX_LEN = 128 # maximum of width of a batch
MAX_PRED = 5  # maxium number predictions to make in a batch
word_to_id, id_to_word = load_vocab('vocab.txt')
VOCAB_SIZE = len(word_to_id)

BATCH_SIZE = 6
n_epochs = 1

n_layers = 6 # number of Encoder of Encoder Layer
n_heads = 12 # number of heads in Multi-Head Attention
d_model = 768 # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2 # number of segments, ex) sentence A and sentence B
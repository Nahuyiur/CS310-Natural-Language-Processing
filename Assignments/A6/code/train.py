import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import time
import sys
from model import BERT
from config import *

random.seed(0)
torch.manual_seed(0)

# Load batchified data
batches_list = pickle.load(open('batches.pkl', 'rb'))
print(f'{len(batches_list)} batches loaded')

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print_every_iter = 10
elapsed_total = 0
elapsed_per_batch = 0
batch_count = 0
for epoch in range(n_epochs):
    for i, batch in enumerate(batches_list):
        start_time = time.time()
        optimizer.zero_grad()
        # Convert batch data to PyTorch tensors
        ### START YOUR CODE ###
        try:
            input_ids = torch.tensor([b[0] for b in batch], dtype=torch.long)
            segment_ids = torch.tensor([b[1] for b in batch], dtype=torch.long)
            masked_pos = torch.tensor([b[3] for b in batch], dtype=torch.long)
            masked_tokens = torch.tensor([b[2] for b in batch], dtype=torch.long)
            is_next = torch.tensor([1 if b[4] else 0 for b in batch], dtype=torch.long)
        except ValueError:
            print(f'batch[{i}] len: {len(batch)}')
            raise
        ### END YOUR CODE ###

        input_ids, segment_ids, masked_tokens, masked_pos, is_next = map(lambda x: x.to(device),
                                                                         [input_ids, segment_ids, masked_tokens, masked_pos, is_next])

        # Forward pass
        ### START YOUR CODE ###
        try:
            logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
            
            # Loss for MLM task
            logits_lm = logits_lm.view(-1, VOCAB_SIZE)
            masked_tokens = masked_tokens.view(-1)
            mask = masked_tokens != 0
            if mask.sum() > 0:
                loss_lm = criterion(logits_lm[mask], masked_tokens[mask])
            else:
                loss_lm = torch.tensor(0.0).to(device)
            
            # Loss for NSP task
            loss_clsf = criterion(logits_clsf, is_next)
            
            # Total loss
            loss = loss_lm + loss_clsf
        except Exception:
            print(f'input_ids: {input_ids}')
            raise
        ### END YOUR CODE ###

        loss.backward()
        optimizer.step()

        end_time = time.time()
        batch_count += 1
        elapsed_total += (end_time - start_time)
        elapsed_per_batch = elapsed_total / batch_count
        # Print
        if i % print_every_iter == 0:
            remain_sec = (len(batches_list) - i) * elapsed_per_batch
            sys.stdout.write(f'\rEpoch {epoch}/{n_epochs}, Batch {i}/{len(batches_list)}, Loss {loss.item():.2f}, Remaining {remain_sec:.2f} sec for current epoch')
            sys.stdout.flush()
    # epoch end
    print()

# Save model
torch.save(model.state_dict(), 'bert_model.pt')
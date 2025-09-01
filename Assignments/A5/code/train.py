import sys
import numpy as np
import datetime
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from get_train_data import FeatureExtractor
from model import BaseModel, WordPOSModel

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_file', default='input_train.npy')
argparser.add_argument('--target_file', default='target_train.npy')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')
argparser.add_argument('--model', default=None, help='path to save model file, if not specified, a .pt with timestamp will be used')


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
    word_vocab_size = len(extractor.word_vocab)
    pos_vocab_size = len(extractor.pos_vocab)
    output_size = len(extractor.rel_vocab)

    ### START YOUR CODE ###
    # TODO: Initialize the model
    # Check if the model is BaseModel or WordPOSModel
    model = None
    if "base" in args.model.lower():
        print("Loading BaseModel...")
        model = BaseModel(word_vocab_size, output_size)
    else:
        print("Loading WordPOSModel...")
        model = WordPOSModel(word_vocab_size, pos_vocab_size ,output_size)
    ### END YOUR CODE ###

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    inputs = np.load(args.input_file)
    targets = np.load(args.target_file) # pytorch input is int
    print("Done loading data.")

    # Train loop
    n_epochs = 10
    batch_size = 10000
    print_loss_every = 50 # every 100 batches

    ### START YOUR CODE ###
    # TODO: Wrap inputs and targets into tensors
    inputs_tensor = torch.from_numpy(inputs).long()
    targets_tensor = torch.from_numpy(targets).long()
    ### END YOUR CODE ###

    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(inputs) // batch_size

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        for batch in dataloader:

            ### START YOUR CODE ###
            # TODO: Get inputs and targets from batch; feed inputs to model and compute loss; backpropagate and update model parameters
            x_batch, y_batch = batch  # Get the batch inputs and targets
            optimizer.zero_grad()  # Zero out the previous gradients
            
            if isinstance(model,WordPOSModel):
                word_indices=x_batch[:, :6]  # 取前6列
                pos_indices=x_batch[:, 6:]   # 取后6列
                inputs=(word_indices,pos_indices)
            else:
                inputs=x_batch
            # Feed inputs to the model
            scores=model(inputs)

            loss = criterion(scores, y_batch)  # Compute the loss using NLLLoss

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters using the optimizer
            ### END YOUR CODE ###

            epoch_loss += loss.item()
            batch_count += 1
            if batch_count % print_loss_every == 0:
                avg_loss = epoch_loss / batch_count 
                sys.stdout.write(f'\rEpoch {epoch+1}/{n_epochs} - Batch {batch_count}/{n_batches} - Loss: {avg_loss:.4f}')
                sys.stdout.flush()
        # print
        print()
        avg_loss = epoch_loss / len(dataloader)
        epoch_end_time = time.time()
        print(f'Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, time: {epoch_end_time - epoch_start_time:.2f} sec')
    
    # save model
    if args.model is not None:
        torch.save(model.state_dict(), args.model)
    else:
        now = datetime.datetime.now()
        torch.save(model.state_dict(), f'model_{now}.pt')
import torch
import torch.nn as nn
from model import BERT
from config import *
import random
import pickle
import sys

random.seed(0)
torch.manual_seed(0)


def load_test_data(file_path):
    test_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            sentence_pair = parts[0]
            is_next = int(parts[1])
            mask_labels = parts[2].split()
            
            # Split into two sentences
            sentences = sentence_pair.split('[SEP]')[:2]
            sentence1 = sentences[0].replace('[CLS]', '').strip() + ' [SEP]'
            sentence2 = sentences[1].strip() + ' [SEP]'
            
            test_data.append({
                'sentence1': sentence1,
                'sentence2': sentence2,
                'is_next': is_next,
                'mask_labels': mask_labels
            })

    return test_data


# A simple space tokenizer
def tokenize(text):
    return text.split()

def make_eval_batch(data, max_pred=MAX_PRED):
    batch = []

    for item in data:
        # Tokenize sentences and add special tokens
        tokens1 = ['[CLS]'] + tokenize(item['sentence1'])
        tokens2 = tokenize(item['sentence2'])
        # Combine tokens and create segment IDs
        tokens = tokens1 + tokens2
        segment_ids = [0] * len(tokens1) + [1] * len(tokens2)
        
        # Find masked positions and their labels
        masked_pos = []
        masked_tokens = []
        mask_labels = item['mask_labels']
        mask_idx = 0
        for i, token in enumerate(tokens):
            if token == '[MASK]' and mask_idx < len(mask_labels):
                masked_pos.append(i)
                masked_tokens.append(mask_labels[mask_idx])
                mask_idx += 1
        
        # Pad masked positions and tokens
        if MAX_PRED > mask_idx:
            n_pad = MAX_PRED - mask_idx 
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        else:
            masked_tokens = masked_tokens[:MAX_PRED]
            masked_pos = masked_pos[:MAX_PRED]

        # Convert to indices
        input_ids = []
        for w in tokens:
            if w in word_to_id:
                input_ids.append(word_to_id[w])
            else:
                input_ids.append(word_to_id['[UNK]'])
        masked_token_ids = []
        for w in masked_tokens:
            if w in word_to_id:
                masked_token_ids.append(word_to_id[w])
            else:
                masked_token_ids.append(word_to_id['[UNK]'])

        # Pad masked_pos to max_pred length
        masked_pos = masked_pos + [0] * (max_pred - len(masked_pos))

        batch.append([input_ids, segment_ids, masked_token_ids, masked_pos, item['is_next']])
    
    # Pad batch to max_len
    max_len = max(len(x[0]) for x in batch)
    for b in batch:
        pad_len = max_len - len(b[0])
        b[0] += [0] * pad_len
        b[1] += [0] * pad_len
    
    return batch


def evaluate(model, eval_batches):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_nsp_correct = 0
    total_mask_correct = 0
    total_mask_tokens = 0
    total_samples = 0
    
    # Process data in batches
    for batch in eval_batches:
        # Convert batch data to tensors
        try:
            input_ids, segment_ids, masked_tokens, masked_pos, is_next = zip(*batch)
            input_ids = torch.LongTensor(input_ids)
            segment_ids = torch.LongTensor(segment_ids)
            masked_tokens = torch.LongTensor(masked_tokens)
            masked_pos = torch.LongTensor(masked_pos)
            is_next = torch.LongTensor(is_next)
        except Exception:
            print(masked_tokens)
            # print(batch)
            raise
        
        # Forward pass
        with torch.no_grad():
            logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
            # NSP accuracy
            nsp_preds = torch.argmax(logits_clsf, dim=1)
            nsp_correct = (nsp_preds == is_next).sum().item()
            total_nsp_correct += nsp_correct
            
            # MLM accuracy
            mask_preds = torch.argmax(logits_lm, dim=2)
            mask_correct = (mask_preds == masked_tokens).sum().item()
            total_mask_correct += mask_correct
            total_mask_tokens += (masked_tokens != 0).sum().item()  # Count non-padded mask tokens
            
            total_samples += len(batch)
    
    nsp_accuracy = total_nsp_correct / total_samples
    mask_accuracy = total_mask_correct / total_mask_tokens if total_mask_tokens > 0 else 0
    
    print(f'NSP Accuracy: {nsp_accuracy:.4f}')
    print(f'Mask Accuracy: {mask_accuracy:.4f}')
    print(f'Total samples: {total_samples}')
    print(f'Total mask tokens: {total_mask_tokens}')
    
    return nsp_accuracy, mask_accuracy


if __name__ == '__main__':
    # Load model
    model_path = sys.argv[1]
    model = BERT()
    model.load_state_dict(torch.load(model_path,
                                     map_location=torch.device('cpu'),
                                     weights_only=True))
    model.eval()
    
    # Evaluate
    eval_batches = pickle.load(open('batches_eval.pkl', 'rb'))
    nsp_acc, mask_acc = evaluate(model, eval_batches)
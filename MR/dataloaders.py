import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse

num_classes = 2

def pad_text(text, pad, min_length=None, max_length=None):
    length = len(text)
    if min_length is not None and length < min_length:
        return text + [pad]*(min_length - length)
    if max_length is not None and length > max_length:
        return text[:max_length]
    return text

class TextDataset(Dataset):
    
    def __init__(self, texts, dictionary, conf_matrix, seed, train=False, val=False, test=False, sort=False, min_length=None, max_length=None):

        PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
        
        self.texts = [([dictionary.indexer(token) for token in text], label) 
                          for text, label in texts]

        if train or val:
            np.random.seed(seed)
            for idx, (tokens, label) in enumerate(self.texts):
                label = int(np.random.choice(num_classes, 1, p=np.array(conf_matrix[label])))
                self.texts[idx] = (tokens, label)

        for idx, (tokens, label) in enumerate(self.texts):
            self.texts[idx] = (idx, tokens, label)

        if min_length or max_length:
            self.texts = [(idx, pad_text(text, PAD_IDX, min_length, max_length), label)
                          for idx, text, label in self.texts]

        if sort:
            self.texts = sorted(self.texts, key=lambda x: len(x[1]))
        
    def __getitem__(self, index):
        idx, tokens, label = self.texts[index]
        return idx, tokens, label
        
    def __len__(self):
        return len(self.texts)
    
class TextDataLoader(DataLoader):
    
    def __init__(self, dictionary, *args, **kwargs):
        super(TextDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
    
    def _collate_fn(self, batch):
        text_lengths = [len(text) for idx, text, label in batch]
        
        longest_length = max(text_lengths)

        idx = [idx for idx, test, label in batch]
        texts_padded = [pad_text(text, pad=self.PAD_IDX, min_length=longest_length) for idx, text, label in batch]
        labels = [label for idx, text, label in batch]

        
        idx_tensor, texts_tensor, labels_tensor = torch.LongTensor(idx), torch.LongTensor(texts_padded), torch.LongTensor(labels)
        return idx_tensor, texts_tensor, labels_tensor

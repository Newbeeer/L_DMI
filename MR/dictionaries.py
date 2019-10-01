import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from os.path import exists

class VDCNNDictionary:
    
    def __init__(self, args):
        
        self.ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/|_#$%^&*~`+=<>()[]{}" # + space, pad, unknown token
        self.PAD_TOKEN = '<PAD>'

    def build_dictionary(self, data):
        
        self.vocab_chars = [self.PAD_TOKEN, '<UNK>', '<SPACE>'] + list(self.ALPHABET)
        self.char2idx = {char:idx for idx, char in enumerate(self.vocab_chars)}
        self.vocabulary_size = len(self.vocab_chars)

    def indexer(self, char):
        if char.strip() == '':
            char = '<SPACE>'
        try:
            return self.char2idx[char]
        except:
            char = '<UNK>'
            return self.char2idx[char]

class CharCNNDictionary:
    
    def __init__(self, args):
        
        self.ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}" + '\n'
        self.PAD_TOKEN = '<PAD>'

    def build_dictionary(self, data):
        
        self.vocab_chars = [self.PAD_TOKEN] + list(self.ALPHABET)
        self.char2idx = {char:idx for idx, char in enumerate(self.vocab_chars)}
        self.vocabulary_size = len(self.vocab_chars)
        self._build_weight()
        
    def _build_weight(self):
        # one hot embedding plus all-zero vector
        onehot_matrix = np.eye(self.vocabulary_size, self.vocabulary_size - 1)
        
        self.embedding = onehot_matrix
        
    def indexer(self, char):
        try:
            return self.char2idx[char]
        except:
            char = self.PAD_TOKEN
            return self.char2idx[char]
        
class AllCharDictionary:
    
    def __init__(self, args):
        
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        if hasattr(args, 'vector_size'):
            self.vector_size = args.vector_size

    def build_dictionary(self, data):
        all_chars = set(char for text, label in data for char in text)
        self.vocab_chars = [self.PAD_TOKEN, self.UNK_TOKEN] + list(sorted(all_chars))
        self.char2idx = {char:idx for idx, char in enumerate(self.vocab_chars)}
        self.vocabulary_size = len(self.vocab_chars)
        self.embedding = None
        
    def indexer(self, char):
        try:
            return self.char2idx[char]
        except:
            char = self.UNK_TOKEN
            return self.char2idx[char]
    
class WordDictionary:
    """
    input : text data, word2vec file
    output : word2idx, weight matrix
    """
    
    def __init__(self, args):
        
        self.max_vocab_size = args.max_vocab_size
        self.min_count = args.min_count
        self.start_end_tokens = args.start_end_tokens
        self.vector_size = args.vector_size
        self.wordvec_mode = args.wordvec_mode
#         self.wordvec_file = args.wordvec_file
        
        self.PAD_TOKEN = '<PAD>'
        
    def build_dictionary(self, data):
        
        self.vocab_words, self.word2idx, self.idx2word = self._build_vocabulary(data)
        self.vocabulary_size = len(self.vocab_words)
        
        if self.wordvec_mode is None:
            self.embedding = None # np.random.randn(self.vocabulary_size, self.vector_size)
        elif self.wordvec_mode == 'word2vec':
            self.embedding = self._load_word2vec()
        elif self.wordvec_mode == 'glove':
            self.embedding = self._load_glove()
            
    def indexer(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']
    
    def _build_vocabulary(self, data):
        
        counter = Counter([word for document, label in data for word in document])
        if self.max_vocab_size:
            counter = {word:freq for word, freq in counter.most_common(self.max_vocab_size)}
        if self.min_count:
            counter = {word:freq for word, freq in counter.items() if freq >= self.min_count}
        
        vocab_words = [self.PAD_TOKEN, '<UNK>']
        
        if self.start_end_tokens: # not necessary for text classification
            vocab_words += ['<SOS>', '<EOS>']
        vocab_words += list(sorted(counter.keys()))
        
        word2idx = {word:idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words # instead of {idx:word for idx, word in enumerate(vocab_words)}
        
        return vocab_words, word2idx, idx2word
    
    def _load_word2vec(self):
        wordvec_file = 'wordvectors/GoogleNews-vectors-negative300.bin'
        if not exists(wordvec_file):
            raise Exception("You must download word vectors through `download_wordvec.py` first")
        word2vec = KeyedVectors.load_word2vec_format(wordvec_file, binary=True)
        self.vector_size = word2vec.vector_size
        
        word_vectors = []
        for word in self.vocab_words:
            
            if word in word2vec.vocab:
                vector = word2vec[word]
            else:
                vector = np.random.normal(scale=0.2, size=self.vector_size) # random vector
            
            word_vectors.append(vector)
        
        weight = np.stack(word_vectors)
        return weight
    
    def _load_glove(self):
        wordvec_file = 'wordvectors/glove.840B.300d.txt'
        self.vector_size = 300
        if not exists(wordvec_file):
            raise Exception("You must download word vectors through `download_wordvec.py` first")

        glove_model = {}
        with open(wordvec_file) as file:
            for line in file:
                line_split = line.split()
                word = ' '.join(line_split[:-self.vector_size])
                numbers = line_split[-self.vector_size:]
                glove_model[word] = numbers
        glove_vocab = glove_model.keys()
                
        word_vectors = []
        for word in self.vocab_words:
            
            if word in glove_vocab:
                vector = np.array(glove_model[word], dtype=float)
            else:
                vector = np.random.normal(scale=0.2, size=self.vector_size) # random vector
            
            word_vectors.append(vector)
        
        weight = np.stack(word_vectors)
        return weight

from os.path import dirname, abspath, join
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

import utils

BASE_DIR = dirname(abspath(__file__))
DATA_DIR = join(BASE_DIR, 'datasets')

class CSVPreprocessor:
    
    def __init__(self, dataset):
        
        dataset_dirname = '{dataset}_csv'.format(dataset=dataset)
        dataset_dir = join(DATA_DIR, dataset_dirname)
        
        self.train_csv = join(dataset_dir, 'train.csv')
        self.test_csv = join(dataset_dir, 'test.csv')
        self.classes_txt = join(dataset_dir, 'classes.txt')

        if dataset in ['yelp_review_polarity', 'yelp_review_full']:
            self.columns = ['class', 'text']
        elif dataset in ['yahoo_answers']:
            self.columns = ['class', 'question_title', 'question_content', 'text']
        else:
            self.columns = ['class', 'title', 'text']
        
    def preprocess(self, level='word', val_size=0.1):
        
        assert level in ['word', 'char'], "level should be either 'word' or 'char'"
        
        train_df = (pd.read_csv(self.train_csv, names=self.columns)
                     .assign(label=lambda x: x['class'].astype(int)-1)
                    )
        self.n_classes = len(train_df['label'].unique())
        
        train_data = self._dataframe_to_data(train_df, level)
        temp = train_data
        train_data, val_data = train_test_split(train_data, test_size=val_size)
        train_data = temp
        test_df = (pd.read_csv(self.test_csv, names=self.columns)
                     .assign(label=lambda x: x['class'].astype(int)-1)
                    )
        test_data = self._dataframe_to_data(test_df, level)
            
        return train_data, val_data, test_data
    
    @staticmethod
    def _dataframe_to_data(dataframe, level):
        dataframe = dataframe.dropna(subset=['text', 'label'])
        if level == 'word':
            dataframe = dataframe.assign(text=lambda df: df['text'].map(lambda text: text.split()))
        elif level == 'char':
            pass
        data = [(text, label) for text, label in zip(dataframe['text'], dataframe['label'])]
        return data

class MRPreprocessor:
    
    def __init__(self, dataset):
        
        dataset_dirname = 'rt-polaritydata'
        dataset_dir = join(DATA_DIR, dataset, dataset_dirname)
        
        self.neg_filepath = join(dataset_dir, 'rt-polarity.neg')
        self.pos_filepath = join(dataset_dir, 'rt-polarity.pos')
        
    def preprocess(self, level, val_size=0.1):
        
        sentences = []
        with open(self.pos_filepath, 'r', errors='ignore') as pos:
            for line in pos:
                cleaned_str = self.clean_str(line)
                words = cleaned_str.split()
                sentences.append((words, 1))
        with open(self.neg_filepath, 'r', errors='ignore') as neg:
            for line in neg:
                cleaned_str = self.clean_str(line)
                words = cleaned_str.split()
                sentences.append((words, 0))
        
        train_data, test_data = train_test_split(sentences, test_size=0.1)
        train_data, val_data = train_test_split(train_data, test_size=0.2)
        self.n_classes = 2

        return train_data, val_data, test_data

    @staticmethod
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

class SSTPreprocessor:
    
    def __init__(self, dataset, binary=False):
        
        self.binary = binary
        dataset_dirname = 'trees'
        dataset_dir = join(DATA_DIR, dataset.upper(), dataset_dirname)
        
        self.train_file = join(dataset_dir, 'train.txt')
        self.val_file = join(dataset_dir, 'dev.txt')
        self.test_file = join(dataset_dir, 'test.txt')
        self.filelist = [self.train_file, self.val_file, self.test_file]
        
        if self.binary:
            self.n_classes = 2
        else:
            self.n_classes = 5
            
    def preprocess(self, level='word'):
        
        tree_dict = defaultdict(list)
        for datafile in self.filelist:
            with open(datafile) as f:
                for line in f:
                    tree = utils.create_tree_from_string(line)
                    for label, line in tree.to_labeled_lines():
                        tree_dict[datafile].append((label, line))
        
        data_dict = defaultdict(list)
        for datafile in self.filelist:
            for label, line in tree_dict[datafile]:
                if self.binary:
                    label = self.polarize(label)
                    if label is None:
                        continue
                if level == 'word':
                    text = line.split()
                else:
                    text = line
                
                data = (text, label)
                data_dict[datafile].append(data)
            
        train_data = data_dict[self.train_file]
        val_data = data_dict[self.val_file]
        test_data = data_dict[self.test_file]
        
        return train_data, val_data, test_data

    @staticmethod
    def polarize(label):
        if label > 2:
            return 1
        elif label < 2:
            return 0
        else: # label == 2
            return None
        
DATASET_TO_PREPROCESSOR = {'ag_news': CSVPreprocessor,
            'amazon_review_full': CSVPreprocessor,
            'amazon_review_polarity': CSVPreprocessor,
            'dbpedia': CSVPreprocessor,
            'sogou_news': CSVPreprocessor,
            'yahoo_answers': CSVPreprocessor,
            'yelp_review_full': CSVPreprocessor,
            'yelp_review_polarity': CSVPreprocessor,
            'MR': MRPreprocessor,
            'SST-1': lambda dataset: SSTPreprocessor('SST', binary=False),
            'SST-2': lambda dataset: SSTPreprocessor('SST', binary=True),
           }

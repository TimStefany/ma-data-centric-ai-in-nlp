import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
import json
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import datasets


class Dataset:
    def __init__(self, dataset_name, data_path='data'):
        self.name = dataset_name
        self.data_path = f'{data_path}/{self.name}/'

        print(f'loading dataset from {self.data_path}')
        try:
            self.info = self.load_info()
            print('dataset info:\n')
            for key, value in self.info.items():
                if key != 'label_text_map':
                    print(f'{key} : {value}')
        except FileNotFoundError:
            print('no <info> file found')
        print('\n')
        
        try:
            self.full = self.generate_tf_dataset_from_csv(self.data_path + 'dataset_full.csv', save_vocab_size=True)
        except FileNotFoundError:
            print('no <full> dataset found')
        try:
            self.train = self.generate_tf_dataset_from_csv(self.data_path + 'train.csv')
        except FileNotFoundError:
            print('no <train> dataset found')
        try:
            self.val = self.generate_tf_dataset_from_csv(self.data_path + 'validation.csv')
        except FileNotFoundError:
            print('no <validation> dataset found')
        try:
            self.test = self.generate_tf_dataset_from_csv(self.data_path + 'test.csv')
        except FileNotFoundError:
            print('no <test> dataset found')
        try:
            test_df = pd.read_csv(self.data_path + 'test.csv')
            self.test_features = test_df['description'].to_numpy()
            self.test_labels = test_df['labels'].to_numpy()
        except FileNotFoundError:
            print('no <test> dataset found')



    def load_info(self):
        with open(self.data_path + 'info.json') as json_file:
            ds_info = json.load(json_file)
        return ds_info
        

    def calculate_vocabulary_size(self, text_column):
        # Create a CountVectorizer object
        vectorizer = CountVectorizer()
        # Fit the vectorizer on the text column
        vectorizer.fit(text_column)
        # Get the vocabulary size
        vocabulary_size = len(vectorizer.vocabulary_)

        return vocabulary_size
    
    def generate_tf_dataset_from_csv(self, csv, save_vocab_size=False):
        # read csv
        df = pd.read_csv(csv)
        if save_vocab_size:
            self.vocab_size = self.calculate_vocabulary_size(df['description'])
        # seperate labels and columns and convert to tensors
        features = df['description']
        labels = df['labels']
        feature_tensor = tf.convert_to_tensor(features.astype(str).values, dtype=tf.string)
        label_tensor = tf.convert_to_tensor(labels.values, dtype=tf.int64)
        # convert label tensor for cathegorical
        label_tensor = tf.one_hot(label_tensor, self.info['num_labels'])
        # create tf_dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((feature_tensor, label_tensor))

        return dataset
    
class TransformerDataset:
    def __init__(self, dataset_name, data_path='data'):
        self.name = dataset_name
        self.data_path = f'{data_path}/{self.name}/'
        
        try:
            test_df = pd.read_csv(self.data_path + 'test.csv')
            self.test_features = test_df['description'].tolist()
            self.test_labels = test_df['labels'].to_numpy()
        except FileNotFoundError:
            print('no <test> dataset found')

        # create training dataset
        print(f'loading dataset from {self.data_path}')
        data_files = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
        self.dataset = datasets.load_dataset(self.data_path, data_files=data_files)

        # load dataset info
        with open(self.data_path + 'info.json') as json_file:
            self.info = json.load(json_file)

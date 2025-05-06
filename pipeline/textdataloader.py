# pipeline/text_dataloader.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset

class TextDataLoader:
    def __init__(self, dataset_name='ag_news', test_size=0.2, random_state=42):
        """
        Args:
            dataset_name (str): 'ag_news', 'sst2', 'trec'
            split_ratio (float)
            random_state (int)
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state

        self.load_data()

    def load_data(self):
        if self.dataset_name == 'ag_news':
            dataset = load_dataset('ag_news')
            texts = dataset['train']['text']
            labels = dataset['train']['label']

        elif self.dataset_name == 'sst2':
            dataset = load_dataset('glue', 'sst2')
            texts = dataset['train']['sentence']
            labels = dataset['train']['label']

        elif self.dataset_name == 'trec':
            dataset = load_dataset('trec')
            texts = dataset['train']['text']
            labels = dataset['train']['coarse_label']

        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported!")

        texts = list(texts)
        labels = list(labels)

        if len(texts) > 500:
            np.random.seed(self.random_state)
            indices = np.random.choice(len(texts), 500, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]

        self.texts = texts
        self.labels = labels

        self.X = np.array(self.texts)
        self.y = np.array(self.labels)

    def split_data(self):
        return train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

    def get_all_data(self):
        return self.X, self.y
    
    def vectorize(self, X_train_raw, X_test_raw):
        """
        Vectorize the raw text data using TF-IDF.
        """
        self.vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = self.vectorizer.fit_transform(X_train_raw).toarray()
        X_test_vec = self.vectorizer.transform(X_test_raw).toarray()
        return X_train_vec, X_test_vec

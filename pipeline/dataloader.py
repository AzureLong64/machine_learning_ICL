# pipeline/dataloader.py
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer

class DataLoader:
    def __init__(self, dataset_name="iris", split_ratio=0.2, random_state=42, stratify=True):
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.stratify = stratify

        self._load_data()

    def _load_data(self):
        if self.dataset_name == "iris":
            data = load_iris()
        elif self.dataset_name == "digits":
            data = load_digits()
        elif self.dataset_name == "wine":
            data = load_wine()
        elif self.dataset_name == "breast_cancer":
            data = load_breast_cancer()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

        self.X = data.data
        self.y = data.target

    def get_data(self):
        return self.X, self.y

    def split_data(self):
        from sklearn.model_selection import train_test_split

        stratify_param = self.y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.split_ratio,
            random_state=self.random_state,
            stratify=stratify_param
        )
        return X_train, X_test, y_train, y_test

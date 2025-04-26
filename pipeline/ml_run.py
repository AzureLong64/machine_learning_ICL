# pipeline/ml_run.py
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class MLRunner:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def run_models(self):
        models = {
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=10000),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }

        model_accuracies = {}
        best_accuracy = 0
        best_model = ""

        for model_name, model in models.items():
            accuracy = self.train_and_evaluate(model)
            print(f"{model_name} Accuracy: {accuracy:.4f}")

            model_accuracies[model_name] = accuracy

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name

        print(f"\nThe best model is: {best_model} with an accuracy of {best_accuracy:.4f}")

        return model_accuracies

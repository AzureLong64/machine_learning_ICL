# contextual_learning.py
import random
import numpy as np
from sklearn.metrics import accuracy_score
from pipeline.client import OpenAI
from tqdm import tqdm

class ContextualLearning:
    def __init__(self, X_train, y_train, X_test, y_test, client=None, 
                 max_test_samples=20, max_train_samples=20):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client = client 
        self.max_test_samples = max_test_samples
        self.max_train_samples = max_train_samples

    def train(self):
        N = min(len(self.X_test), self.max_test_samples)
        true_label = []
        pred_label = []

        print(f"\nUsing {N} test samples and {self.max_train_samples} train samples per prompt.")

        for n in tqdm(range(N), desc="Contextual Learning Inference Progress"):
            selected_indices = np.random.choice(len(self.X_train), 
                                                size=min(self.max_train_samples, len(self.X_train)), 
                                                replace=False)
            X_train_selected = self.X_train[selected_indices]
            y_train_selected = self.y_train[selected_indices]

            prompt = "Help me predict the Output value for the last Input. Your response should only contain the Output value in the format of #Output value#.\n"

            for i in range(len(X_train_selected)):
                prompt += f"Input: {X_train_selected[i]}, Output: {y_train_selected[i]}\n"
            prompt += f"Input: {self.X_test[n]}, Output: "

            max_tries = 5
            err_counter = 0
            #print(prompt)
            while err_counter < max_tries:
                try:
                    completion = self.client.chat.completions.create(
                        model='deepseek-chat', 
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0
                    )
                    response = completion.choices[0].message.content
                    pred = int(response.replace("#", ""))
                    # print(pred)
                    # print(self.y_test[n])

                    break
                except Exception as e:
                    err_counter += 1

            if err_counter == max_tries:
                pred = random.randint(0, 9)

            true_label.append(self.y_test[n])
            pred_label.append(pred)

        accuracy = accuracy_score(true_label, pred_label)
        print(f"\nContextual Learning Model Accuracy: {accuracy:.4f}")
        return accuracy

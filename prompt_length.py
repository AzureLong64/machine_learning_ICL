# 本模块用于实验prompt长度（用于训练的样例数量）对ICL的准确率的影响
# 运行后生成results/icl_prompt_effect.csv与results/icl_prompt_effect.png
# Defalut configs:
#   Runs on dataset 'ag_news'
#   llm: DeepSeek
import os
import pandas as pd
import matplotlib.pyplot as plt  
from pipeline.textdataloader import TextDataLoader
from pipeline.client import APIClient
from pipeline.contextual_learning import ContextualLearning

def main():
    api_wrapper = APIClient(api_key="sk-4ac41f95e3414ca78790f1af126c02e4")
    print("Loading llm...")
    client = api_wrapper.get_client()
    models = client.models.list()
    print("Model loaded successfully:\n", models)
    print("============================================")

    dataset_name = "ag_news"  
    print(f"=== Running on dataset: {dataset_name} ===")

    data_loader = TextDataLoader(dataset_name, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = data_loader.split_data()

    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print("=========================================================")
    print("Start to run contextual learning...")

    start_train_samples = 5
    max_train_samples = 50
    step = 5

    train_samples_list = []
    accuracy_list = []

    for train_samples in range(start_train_samples, max_train_samples + step, step):
        print(f"Running ICL with max_train_samples = {train_samples}")
        contextual_learning = ContextualLearning(
            X_train, y_train, X_test, y_test,
            client=client,
            model_name='deepseek-chat',
            max_test_samples=20,  
            max_train_samples=train_samples
        )
        icl_accuracy = contextual_learning.train()
        print(f"Accuracy with {train_samples} training samples: {icl_accuracy:.4f}")

        train_samples_list.append(train_samples)
        accuracy_list.append(icl_accuracy)

    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame({
        "Train_Samples": train_samples_list,
        "Accuracy": accuracy_list
    })
    results_df.to_csv("results/icl_prompt_effect.csv", index=False)
    print("Saved results to results/icl_prompt_effect.csv!")

    plt.figure(figsize=(8, 6))
    plt.plot(train_samples_list, accuracy_list, marker='o', linestyle='-', color='b')
    plt.title(f'Effect of Train Samples on ICL Accuracy ({dataset_name})')
    plt.xlabel('Number of Train Samples')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig("results/icl_prompt_effect.png") 
    plt.show()
    print("Saved curve to results/icl_prompt_effect.png!")

if __name__ == "__main__":
    main()

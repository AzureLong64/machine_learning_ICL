# 本模块用于实验不同大模型对ICL准确率的影响
# 输出：results/icl_model_compare.csv 与 results/icl_model_compare.png

import os
import pandas as pd
import matplotlib.pyplot as plt
from pipeline.textdataloader import TextDataLoader
from pipeline.contextual_learning import ContextualLearning
from openai import OpenAI


def main():
    model_names = [
        'Qwen/Qwen2-1.5B-Instruct',
        'Qwen/Qwen2-7B-Instruct',
        'deepseek-chat'
    ]

    API_KEY = "sk-ganzjcgtiqzszibsfiqenqikpwgatojlsfvtmrcvvqnpbzzl"
    ds_api = "sk-4ac41f95e3414ca78790f1af126c02e4"
    client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
    ds_client = OpenAI(api_key=ds_api, base_url="https://api.deepseek.com")

    dataset_name = "ag_news"
    print(f"======Running ICL comparison on dataset: {dataset_name}======")

    data_loader = TextDataLoader(dataset_name, split_ratio=0.2, random_state=42)
    X_train, X_test, y_train, y_test = data_loader.split_data()

    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    model_list = []
    accuracy_list = []

    for model_name in model_names:
        if model_name == 'deepseek-chat':
            print(f"\nRunning ICL with model: {model_name}")
            contextual_learning = ContextualLearning(
                X_train, y_train, X_test, y_test,
                client=ds_client,
                model_name=model_name,
                max_test_samples=20,
                max_train_samples=20
            )
            icl_accuracy = contextual_learning.train()
            print(f"Model: {model_name} | Accuracy: {icl_accuracy:.4f}")

            model_list.append(model_name)
            accuracy_list.append(icl_accuracy)
        else:
            print(f"\nRunning ICL with model: {model_name}")
            contextual_learning = ContextualLearning(
                X_train, y_train, X_test, y_test,
                client=client,
                model_name=model_name,
                max_test_samples=20,
                max_train_samples=20
            )
            icl_accuracy = contextual_learning.train()
            print(f"Model: {model_name} | Accuracy: {icl_accuracy:.4f}")

            model_list.append(model_name)
            accuracy_list.append(icl_accuracy)

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame({
        "Model": model_list,
        "Accuracy": accuracy_list
    })
    df.to_csv("results/icl_model_compare.csv", index=False)
    print("Saved results to results/icl_model_compare.csv")

    plt.figure(figsize=(10, 6))
    plt.bar(model_list, accuracy_list, color='skyblue')
    plt.title(f'ICL Accuracy Comparison across LLMs ({dataset_name})')
    plt.ylabel('Accuracy')
    plt.xlabel('LLM Model')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("results/icl_model_compare.png")
    plt.show()
    print("Saved chart to results/icl_model_compare.png!")


if __name__ == "__main__":
    main()

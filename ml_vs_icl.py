import os
import pandas as pd
import matplotlib.pyplot as plt
from pipeline.dataloader import DataLoader
from pipeline.textdataloader import TextDataLoader
from pipeline.client import APIClient
from pipeline.ml_run import MLRunner
from pipeline.contextual_learning import ContextualLearning

def main():
    # Load LLM client
    api_wrapper = APIClient(api_key="sk-4ac41f95e3414ca78790f1af126c02e4")
    print("Loading llm...")
    client = api_wrapper.get_client()
    models = client.models.list()
    print("Model loaded successfully:\n", models)
    print("============================================")

    # Define datasets
    structured_datasets = ["iris", "wine"]
    text_datasets = ["ag_news", "sst2"]
    datasets = structured_datasets + text_datasets

    results = []

    for dataset_name in datasets:
        print("Loading and splitting data...")
        print(f"=== Running on dataset: {dataset_name} ===")

        if dataset_name in structured_datasets:
            data_loader = DataLoader(dataset_name, test_size=0.2, random_state=42)
            X_train, X_test, y_train, y_test = data_loader.split_data()
            X_train_for_icl, X_test_for_icl = X_train, X_test
        else:
            text_data_loader = TextDataLoader(dataset_name, test_size=0.2, random_state=42)
            X_train_raw, X_test_raw, y_train, y_test = text_data_loader.split_data()
            X_train, X_test = text_data_loader.vectorize(X_train_raw, X_test_raw)
            X_train_for_icl, X_test_for_icl = X_train_raw, X_test_raw

        print(f"Train data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print("=========================================================")

        # Run traditional ML models
        print("Start to run traditional machine learning methods...")
        ml_runner = MLRunner(X_train, X_test, y_train, y_test)
        ml_results = ml_runner.run_models()
        print("=========================================================")

        # Run ICL
        print("Start to run contextual learning...")
        contextual_learning = ContextualLearning(
            X_train_for_icl, y_train, X_test_for_icl, y_test,
            client=client,
            model_name='deepseek-chat'
        )
        icl_accuracy = contextual_learning.train()
        print("=========================================================")

        # Save results
        for model_name, acc in ml_results.items():
            results.append({
                "Dataset": dataset_name,
                "Method": model_name,
                "Accuracy": acc
            })

        results.append({
            "Dataset": dataset_name,
            "Method": "ICL",
            "Accuracy": icl_accuracy
        })

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/results.csv", index=False)
    print("All results saved to results/results.csv!")

    for dataset in datasets:
        df_dataset = df[df["Dataset"] == dataset]
        plt.figure(figsize=(8, 6))
        plt.bar(df_dataset["Method"], df_dataset["Accuracy"], color="skyblue")
        plt.title(f"Accuracy Comparison on {dataset}")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=30)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"results/accuracy_comparison_{dataset}.png")
        plt.close()

    print("Saved accuracy comparison plots to results/*.png!")

if __name__ == "__main__":
    main()

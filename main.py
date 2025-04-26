# main.py
import os
from pipeline.dataloader import DataLoader
from pipeline.textdataloader import TextDataLoader
from pipeline.client import APIClient
from pipeline.ml_run import MLRunner
from pipeline.contextual_learning import ContextualLearning
import pandas as pd

def main():
    # load client
    # default: deepseek
    api_wrapper = APIClient(api_key="sk-db73e90e0b1e4281873826504cb085a4")
    print("Loading llm...")
    client = api_wrapper.get_client()
    models = client.models.list()
    print("Model loaded successfully:\n", models)
    print("============================================")

    # get data
    structured_datasets = ["iris", "digits", "wine"]
    text_datasets = ["ag_news", "sst2", "trec"]
    datasets = structured_datasets + text_datasets

    results = []
    for dataset_name in datasets:
        print("Loading and splitting data...")
        print(f"=== Running on dataset: {dataset_name} ===")
        
        if dataset_name in structured_datasets:
            data_loader = DataLoader(dataset_name, split_ratio=0.2, random_state=42)
            X_train, X_test, y_train, y_test = data_loader.split_data()
            X_train_for_icl, X_test_for_icl = X_train, X_test
        else:
            data_loader = TextDataLoader(dataset_name, split_ratio=0.2, random_state=42)
            X_train_raw, X_test_raw, y_train, y_test = data_loader.split_data()
            X_train, X_test = data_loader.vectorize(X_train_raw, X_test_raw)
            X_train_for_icl, X_test_for_icl = X_train_raw, X_test_raw

        print(f"Train data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print("=========================================================")

        print("Start to run traditional machine learning methods...")
        ml_runner = MLRunner(X_train, X_test, y_train, y_test)
        ml_results = ml_runner.run_models()
        print("=========================================================")

        # Start contextual learning
        print("Start to run contextual learning...")
        contextual_learning = ContextualLearning(X_train_for_icl, y_train, X_test_for_icl, y_test, client=client)
        icl_accuracy = contextual_learning.train()
        print("=========================================================")

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
    df.to_csv("results.csv", index=False)
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("results/results.csv", index=False)  
    print("All results saved to results/results.csv!")


if __name__ == "__main__":
    main()

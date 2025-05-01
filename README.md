# machine_learning_ICL
This project is aim to compare the traditional machine learning algorithms and the In-context learning(ICL)
The project is divided into two parts:
1. Explore what kind of tasks/datasets is ICL better/worse than classical supervised learning methods.
2. Explore how to improve the performance of ICL(i.e change the size/prompt/tempreture of the LLM).
machine_learning_ICL/
│
├── pipeline/
│   ├── client.py              # Initialize the LLM client
│   ├── contextual_learning.py # Run ICL inference
│   ├── dataloader.py          # Load structured datasets
│   ├── textdataloader.py      # Load and vectorize text datasets
│   ├── ml_run.py              # Run traditional ML algorithms (SVM, RF, etc.)
│   ├── prompt_generator.py    # [TODO] Improve prompts for ICL
│
├── results/                   # Store experiment results (CSV, plots, etc.)
│
├── main.py                    # Entry point: run experiments comparing ML vs. ICL

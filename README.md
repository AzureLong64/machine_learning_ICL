# machine_learning_ICL
This project is aim to compare the traditional machine learning algorithms and the In-context learning(ICL)
The project is divided into two parts:
1. Explore what kind of tasks/datasets is ICL better/worse than classical supervised learning methods.
2. Explore how to improve the performance of ICL(i.e change the size/prompt/tempreture of the LLM).

\pipeline
    \client.py                  initialize the llm
    \contexual_learning.py      run ICL
    \dataloader                 load and pre-process the data
    \ml_run                     run the traditional machine learning methods
    \prompt_generator           to improve the prompt ###TODO
    \textdataloader             load the text_data and vectorize
\results
\main.py
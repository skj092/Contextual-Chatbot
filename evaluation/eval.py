from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
import pandas as pd
import os
import requests
from tqdm import tqdm
import ast
import json

test_data = "/home/sonujha/rnd/qp-ai-assessment/data/testset.csv"
log_file_path = "data/experiment_log.csv"

# os.environ["OPENAI_API_KEY"] = "your-openai-key"

# format
"""
questions = ['question1', 'question2'...'questionN'] - shape (N,)
answers = ['answer1', 'answer2'...'answerN'] - shape (N,)
contexts = [['context1_1', 'context1_2'...'context1_K'], ['context2_1', 'context2_2'...'context2_K']...['contextN_1', 'contextN_2'...'contextN_K']] - shape (N, K)
ground_truth = ['ground_truth1', 'ground_truth2'...'ground_truthN'] - shape (N,)
"""

data_samples = {
    "question": ["When was the first super bowl?", "Who won the most super bowls?"],
    "answer": [
        "The first superbowl was held on Jan 15, 1967",
        "The most super bowls have been won by The New England Patriots",
    ],
    "contexts": [
        [
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
        ],
        [
            "The Green Bay Packers...Green Bay, Wisconsin.",
            "The Packers compete...Football Conference",
        ],
    ],
    "ground_truth": [
        "The first superbowl was held on January 15, 1967",
        "The New England Patriots have won the Super Bowl a record six times",
    ],
}


# Get the answer from the api
def get_answer(question):
    url = "http://0.0.0.0:52207/query"
    answer = requests.post(url, json={"text": question}).json()
    return answer


def get_answer_of_question(test_data):
    df = pd.read_csv(test_data)
    question_list = df["question"].tolist()
    answer_list = []
    for question in tqdm(question_list):
        answer = get_answer(question)
        answer_list.append(answer["response"])
    df["answer"] = answer_list
    df.to_csv("data/question_answer.csv")
    return df


if os.path.exists(test_data):
    df = get_answer_of_question(test_data)

    contexts = df["contexts"].tolist()
    contexts = [
        ast.literal_eval(item) if isinstance(item, str) else item for item in contexts
    ]
    data_samples = {
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": contexts,
        "ground_truth": df["ground_truth"].tolist(),
    }
dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
score = score.to_pandas()
score.to_csv("data/score.csv")


def update_log_file(log_file):
    with open("config.json") as f:
        config = json.load(f)
    df = pd.DataFrame([config])
    if os.path.exists(log_file_path):
        log_file = pd.read_csv(log_file_path)
        df = pd.concat([log_file, df])
        print(f"log file {log_file} updated")
    df["answer_correctness"] = score["answer_correctness"].mean()
    df.to_csv("data/experiment_log.csv")


print("average answer_correctness: ", score["answer_correctness"].mean())
update_log_file(log_file_path)

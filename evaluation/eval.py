from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
import pandas as pd
import os
import requests
from tqdm import tqdm
import ast

test_data = "/home/sonujha/rnd/qp-ai-assessment/testset.csv"

# os.environ["OPENAI_API_KEY"] = "your-openai-key"

# format
'''
questions = ['question1', 'question2'...'questionN'] - shape (N,)
answers = ['answer1', 'answer2'...'answerN'] - shape (N,)
contexts = [['context1_1', 'context1_2'...'context1_K'], ['context2_1', 'context2_2'...'context2_K']...['contextN_1', 'contextN_2'...'contextN_K']] - shape (N, K)
ground_truth = ['ground_truth1', 'ground_truth2'...'ground_truthN'] - shape (N,)
'''

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts': [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'],
                 ['The Green Bay Packers...Green Bay, Wisconsin.', 'The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}


# Get the answer from the api
def get_answer(question):
    url = "http://127.0.0.1:8000/query"
    answer = requests.post(url, json={"text": question}).json()
    return answer


def get_answer_of_question(test_data):
    df = pd.read_csv(test_data)
    df = df.sample(n=2)
    question_list = df['question'].tolist()
    answer_list = []
    for question in tqdm(question_list):
        answer = get_answer(question)
        answer_list.append(answer['response'])
    df['answer'] = answer_list
    df.to_csv("question_answer.csv")
    return df


if os.path.exists(test_data):
    if os.path.exists('question_answer.csv'):
        df = pd.read_csv('question_answer.csv')
    else:
        df = get_answer_of_question(test_data)

    contexts = df['contexts'].tolist()
    contexts = [ast.literal_eval(item) if isinstance(
        item, str) else item for item in contexts]
    data_samples = {
        'question': df['question'].tolist(),
        'answer': df['answer'].tolist(),
        'contexts': contexts,
        'ground_truth': df['ground_truth'].tolist()
    }
dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
score = score.to_pandas()
score.to_csv('score.csv')

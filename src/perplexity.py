from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)


def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results


def measure_test_perplexity(model, tokenizer, data):
    perplexity_scores = []

    for row in data["text"]:
        row = row[:300]  # Considering the first 100 tokens for simplicity
        inputs = tokenizer(row, return_tensors="pt")
        loss = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss
        ppl = torch.exp(loss)
        ppl = ppl.item()
        perplexity_scores.append(ppl)

    return perplexity_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)
    parser.add_argument("--feat_perplexity", "-fperp", help="Feature perplexity", default=True)

    args = parser.parse_args()

    random_seed = 0
    train_path =  args.train_file_path # For example 'subtaskA_train_multilingual.jsonl'
    test_path =  args.test_file_path # For example 'subtaskA_test_multilingual.jsonl'
    model =  args.model # For example 'xlm-roberta-base'
    subtask =  args.subtask # For example 'A'
    prediction_path = args.prediction_file_path # For example subtaskB_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)

    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)
    
    
    if args.feat_perplexity:
        perp_model = AutoModelForCausalLM.from_pretrained("gpt2")
        perpl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        train_df = train_df.head(10000)
        valid_df = valid_df.head(3000)
        perplexity_scores = measure_test_perplexity(perp_model, perpl_tokenizer, train_df)

        features = np.array(perplexity_scores).reshape(-1, 1)


        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(features, train_df['label'])

        valid_perplexity_scores = measure_test_perplexity(perp_model, perpl_tokenizer, valid_df)
        valid_features = np.array(valid_perplexity_scores).reshape(-1, 1)

        pred = svm_classifier.predict(valid_features)
        
        
        print(classification_report( valid_df['label'], pred))
  
        
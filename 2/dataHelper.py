import datasets
import os
import json
import jsonlines
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

dataset_class_num = {
    'restaurant_sup': 3,
    'agnews_sup': 4,
    'acl_sup': 6
}

def ABSA(name):
    name = name.split('_')[0]
    def json_transform(type):
        with open(f'./{name}_sup/{type}.json', 'r') as f:
            original_data = json.load(f)

        transformed_data = [value for key, value in original_data.items()]

        with open(f'./{name}_sup/{type}_new.json', 'w') as f:
            json.dump(transformed_data, f, indent=4)

    json_transform('train')
    json_transform('test')
    train_dataset = load_dataset('json', data_files=f'./{name}_sup/train_new.json', split='train')
    test_dataset = load_dataset('json', data_files=f'./{name}_sup/test_new.json', split='train')
    dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
    })
    sep_token = ''
    label2idx = {'positive': 0, 'neutral': 1, 'negative': 2}
    def get_item(example):
        example['text'] = example['term'] + sep_token + example['sentence']
        example['label'] = label2idx[example['polarity']]
        return example

    dataset = dataset.map(get_item)
    dataset = dataset.select_columns(['text', 'label'])
    return dataset, label2idx

def acl_sup():
    dataset = load_dataset('json', data_files={'train': './acl_sup/train.jsonl', 'test': './acl_sup/test.jsonl'})
    label2idx = {'CompareOrContrast': 0, 'Background': 1, 'Extends': 2, 'Future': 3, 'Motivation': 4, 'Uses':5}
    dataset = dataset.map(lambda example: {'label': label2idx[example['label']]}, remove_columns='metadata')
    return dataset, label2idx

def agnews_sup():
    dataset = load_dataset('csv', data_files='agnews_sup/agnews_sup.csv', column_names=['label', 'title', 'description'], split='train')
    dataset = dataset.map(lambda example: {'text': example['description'], 'label': example['label'] - 1}, remove_columns=['title', 'description'])
    dataset = dataset.train_test_split(test_size=0.1, seed=2022)
    label2id = {'1': 0, '2': 1, '3': 2, '4': 3}
    return dataset, label2id

def few_shot_version(name):
    func_map = {
        'restaurant_fs': ABSA,
        'laptop_fs': ABSA,
        'acl_fs': acl_sup,
        'agnews_fs': agnews_sup
    }
    func = func_map.get(name)
    if func == acl_sup or func == agnews_sup:
        dataset = func()['train']
    elif func == ABSA:
        dataset = func(name)['train']
    else:
        raise ValueError("Invalid name provided")
    label_num = len(set(dataset['label']))
    avg_num = int(32/label_num) if label_num < 5 else 8
    selected_examples = {}
    for label in range(label_num):
        filtered_dataset = dataset.filter(lambda example: example['label'] == label)
        filtered_dataset = filtered_dataset.shuffle(seed=42)
        selected_examples[label] = filtered_dataset.select(range(avg_num))
    dataset = concatenate_datasets([selected_examples[label] for label in range(label_num)])
    return dataset

def get_dataset(dataset_names):
    datasets = []
    label2ids = {}
    current_label = 0
    for dataset_name in dataset_names:
        data_type = dataset_name.split('_')[-1]
        if data_type == 'fs':
            datasets.append(few_shot_version(dataset_name))
        else:
            func_map = {
                'restaurant_sup': ABSA,
                'laptop_sup': ABSA,
                'acl_sup': acl_sup,
                'agnews_sup': agnews_sup
            }
            func = func_map.get(dataset_name)
            if func == acl_sup or func == agnews_sup:
                dataset,  label2id= func()
            elif func == ABSA:
                dataset, label2id= func(dataset_name)
            datasets.append(dataset)
            label2id_new = {key: value + current_label for key, value in label2id.items()}
            label2ids.update(label2id_new)
            current_label += dataset_class_num.get(dataset_name)
    train_dataset = concatenate_datasets([dataset['train'] for dataset in datasets])
    test_dataset = concatenate_datasets([dataset['test'] for dataset in datasets])
    dataset = DatasetDict(
        {
            'train': train_dataset,
            'test': test_dataset
        }
    )
    return dataset, label2ids

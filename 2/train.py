from dataHelper import get_dataset
import evaluate
import numpy as np
import os
import torch
from datasets import load_dataset
import datasets
import transformers
from transformers import(
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field
import logging
import sys
from sklearn.metrics import f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = 'train'
os.environ['WANDB_LOG_MODEL'] = 'checkpoint'

#set up HfArgumentParser
@dataclass
class MethodAndDataset: 
    model_name: str = field(
    default = 'bert-base-uncased',
    metadata = {'help': 'Select a pretrained model, you can choose between roberta-base, bert-base-uncased and allenai/scibert_scivocab_uncased'}
    )
    dataset_name: str = field(
    default = 'acl_sup',
    metadata = {'help': 'Select a dataset, you can choose from restaurant_sup, acl_sup and agnews_sup'}
    )
parser = HfArgumentParser(MethodAndDataset)
args = parser.parse_args_into_dataclasses()
print(f'model: {args[0].model_name}')
print(f'dataset: {args[0].dataset_name}')
#set up logging
datasets.utils.logging.set_verbosity_info()
transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_default_handler()
#Enable the default handler of the HuggingFace Transformers’s root logger.
transformers.utils.logging.enable_explicit_format()
#Enable explicit formatting for every HuggingFace Transformers’s logger.


set_seed(2024)

#get dataset
dataset_names = args[0].dataset_name.split(',')
dataset, label2id = get_dataset(dataset_names)

#set labels
id2label = {value: key for key, value in label2id.items()}

#load pretrained model and tokenizer
model_name = args[0].model_name
num_labels = len(label2id)
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
config.id2label = id2label
config.label2id = label2id
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                           config=config)


#process dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)
tokenized_data = dataset.map(preprocess_function, batched=True)

#evaluate
accuracy = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {
    'accuracy': accuracy.compute(predictions=predictions, references=labels)['accuracy'],
    'micro_f1': micro_f1,
    'macro_f1': macro_f1
    }

#training parameters
training_args = TrainingArguments(
    output_dir="acl_bert_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to='wandb',
    logging_steps=1,
)
 
#train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
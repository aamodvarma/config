from transformers import GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, DataCollatorForLanguageModeling, AutoTokenizer
import torch

OUTPUT_DIR = "./models/model_wordlevel_epoch50"

def make_dataset(ty, datafile):
    dataset = load_dataset(ty, data_files=datafile, split="train")
    return dataset


dataset = make_dataset("text", "data/composition.txt")
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizers/WordLevelTokenizer")


# Load the data
# dataset = load_dataset("text", data_files="data/composition.txt")

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_dataset(dataset):
    dataset_with_end = [text + f" [END]" for text in dataset['text']]
    return tokenizer(dataset_with_end)
tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
tokenized_datasets = tokenized_dataset.remove_columns(["text"])



data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch"
)


import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train();
model.save_pretrained(OUTPUT_DIR)

from transformers import GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, DataCollatorForLanguageModeling, AutoTokenizer
import torch
import numpy as np

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
    return tokenizer(dataset["text"], padding=True)
tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
criterion = torch.nn.CrossEntropyLoss()
i = 0


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

starting_vector = torch.randn(768).unsqueeze(0);
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}\
        input_ids = batch['input_ids']
        input_embeddings = model.transformer.wte(input_ids)

        for i in range(len(input_embeddings)):
            input_embeddings[i] = torch.cat([starting_vector, input_embeddings[i]], dim=0)[0:20]

        outputs = model(inputs_embeds=input_embeddings)


        
        # outputs = model(**batch)
        logits = outputs.logits
        loss = criterion(logits.view(-1, logits.size(-1)), batch["input_ids"].view(-1))

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

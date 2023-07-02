import os 
os.chdir('..')

import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from collections import defaultdict
import numpy as np

import os

directory = "./gpt2_decoder_baseline"

if not os.path.exists(directory):
    os.makedirs(directory)


paths = [str(x) for x in Path("./datasets/babylm_10M/").glob("*.train")]


tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
])

tokenizer.enable_truncation(max_length=512)

tokenizer.save_model(directory)


from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, AutoConfig

tokenizer = GPT2TokenizerFast.from_pretrained(directory, max_len=512)

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=128,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


model = GPT2LMHeadModel(config)

print('num of params:', model.num_parameters())


from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./datasets/babylm_10M_merged.train",
    block_size=128,
)


from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)


from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir=directory,
    overwrite_output_dir=True,
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    num_train_epochs=10,
    save_steps=2000,
    save_total_limit=20,
    seed=12,
#     evaluate_during_training=True,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(directory)



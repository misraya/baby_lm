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

directory = "./blenderbot_small_baseline"

if not os.path.exists(directory):
    os.makedirs(directory)




paths = [str(x) for x in Path("./datasets/babylm_10M/").glob("*.train")]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])


tokenizer.save_model(directory)


from tokenizers.implementations import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer(
    directory+"/vocab.json",
    directory+"/merges.txt",
)

from tokenizers.processors import BertProcessing

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)


from transformers import BlenderbotSmallConfig

config = BlenderbotSmallConfig(
    vocab_size=52_000,
)



from transformers import BlenderbotSmallTokenizer

tokenizer = BlenderbotSmallTokenizer.from_pretrained(directory, max_len=512)


from transformers import  BlenderbotSmallForConditionalGeneration 

model =  BlenderbotSmallForConditionalGeneration(config=config)


print(model.num_parameters())


from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./datasets/babylm_10M_merged.train",
    block_size=128,
)


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=directory,
    overwrite_output_dir=True,
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    num_train_epochs=10,
    save_steps=10000,
    save_total_limit=10,
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






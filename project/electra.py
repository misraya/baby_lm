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
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import ByteLevelBPETokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


import os

directory = "./my_baselines/electra_baseline"

if not os.path.exists(directory):
    os.makedirs(directory)


paths = [str(x) for x in Path("./datasets/babylm_10M/").glob("*.train")]

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

tokenizer.pre_tokenizer = Whitespace()

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $0 [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
)

trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.train(paths, trainer)


from transformers import ElectraConfig

config = ElectraConfig(is_decoder=True)


from transformers import ElectraTokenizerFast

transformer_tokenizer = ElectraTokenizerFast(tokenizer_object=tokenizer)

transformer_tokenizer.save_pretrained(directory)


from transformers import ElectraForCausalLM

model = ElectraForCausalLM(config=config)


print(model.num_parameters())


from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=transformer_tokenizer,
    file_path="./datasets/babylm_10M_merged.train",
    block_size=128,
)


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=transformer_tokenizer, mlm=True, mlm_probability=0.15
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






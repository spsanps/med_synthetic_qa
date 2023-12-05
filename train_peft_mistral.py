# %%

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Check if --inv flag is set.')

# Add the --inv flag
# The action 'store_true' will set the variable to True if --inv is used in the command line
parser.add_argument('--inv', action='store_true', help='Activate the inv flag')

# Parse the arguments
args = parser.parse_args()

# Set a boolean based on the presence of the --inv flag
is_inv_active = args.inv

import pandas as pd 

# read df.to_csv('./data/training_data.csv', index=False)

df = pd.read_csv('./data/training_data.csv')
df.head()

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "ehartford/dolphin-2.2.1-mistral-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
#bnb_config = BitsAndBytesConfig(
#    load_in_8bit=True,
#)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

# %%
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    max_length=1024,
    padding="max_length",
    #add_eos_token=True,
    #add_bos_token=True,
    truncation=True,
)
tokenizer.pad_token = tokenizer.eos_token

# %%
from datasets import Dataset

# Assuming 'df' is your pandas DataFrame and it has columns 'qa_data' and 'para_data'
# Separate the data into two dictionaries

import random
random.seed(42)

para_data_sel = df["para_data"].tolist()
qa_data_sel = df["qa_data"].tolist()

n = 20000

para_data_sel = para_data_sel[:n]
qa_data_sel = qa_data_sel[:n]

val_n = 2000

if not is_inv_active:
    train, val = qa_data_sel, para_data_sel
else:
    train, val = para_data_sel, qa_data_sel

val = random.sample(val, val_n)

#shuffle train

random.shuffle(train)

train_data = {
    "text": train, # Convert the 'qa_data' column to a list
}

val_data = {
    "text": val, # Convert the 'para_data' column to a list
}

# Create two separate Hugging Face Dataset objects
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

#print(train_dataset[0])
#print(val_dataset[0])

# %%
def generate_and_tokenize_prompt(prompt):
    return tokenizer(prompt["text"], truncation=True, padding="max_length", max_length=1024)

#train_dataset, val_dataset = val_dataset, train_dataset

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = val_dataset.map(generate_and_tokenize_prompt)

# %%
print(tokenized_val_dataset[0])
print(len(tokenized_val_dataset[0]["input_ids"]))

# %%
import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

# %%
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# %%
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print(model)

# %%
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# %%
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model = accelerator.prepare_model(model)

# %%
import wandb, os
wandb.login()


if not is_inv_active:
    wandb_project = "medqa_mixed_qa_train_20k"
else:
    wandb_project = "medqa_mixed_para_train_20k"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# %%
model.to("cuda")


# %%
import transformers
from datetime import datetime

project = wandb_project
base_model_name = "medqa_mistral_dolphin_2_2_1"
run_name = base_model_name + "-" + project
output_dir = "./train_outputs/" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=150,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=1,
        learning_rate=1e-6, # Want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=250,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=250,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# %%

# python train_peft_mistral.py --inv > logs/train_peft_mistral_inv2.log 2>&1 &
# python train_peft_mistral.py > logs/train_peft_mistral2.log 2>&1 &




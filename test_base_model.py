# load full_dataset/data_clean/questions/US/test.jsonl

import json
import os
import pandas as pd

# load data
with open('full_dataset/data_clean/questions/US/test.jsonl') as f:
    data = f.readlines()
data = [json.loads(line) for line in data]

# convert to dataframe
df = pd.DataFrame(data)
df.head()


# load model
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

# Re-init the tokenizer so it doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    # padding_side="left",
)


sys_prompt = """
You are an extremely skilled doctor, who will accurately respond to medical questions exactly as the user asks.
""".strip()

user_template = """

Please reason generally about answer to the following question:

{question}""".strip()

prompt_template = """<|im_start|>system
 {sys_prompt} <|im_end|>
<|im_start|>user
 {user_prompt} <|im_end|>
<|im_start|>assistant
 """

def options_to_string(options_dict):
    options = []
    for k, v in options_dict.items():
        options.append(f"{k}: {v}")
    return "\n".join(options)


def make_prompt(dataset_row):
    question = dataset_row["question"]
    options = dataset_row["options"]

    # print(dataset_row["answer_idx"])

    options = options_to_string(options)

    question = f"{question}\n{options}"

    prompt = prompt_template.format(
        sys_prompt=sys_prompt,
        user_prompt=user_template.format(question=question)
    )

    return prompt


# if progress.txt exists, load it
if os.path.exists("progress.txt"):
    with open("progress.txt", "r") as f:
        progress = int(f.read())
else:
    progress = 0

print(f"Starting at {progress}")

from tqdm import tqdm

import json
import jsonlines


# if prompts.jsonl and outputs.jsonl exist
if os.path.exists("prompts.jsonl") and os.path.exists("outputs.jsonl"):
    prompts = []
    outputs = []
    with jsonlines.open("prompts.jsonl") as f:
        for line in f:
            prompts.append(line)
    with jsonlines.open("outputs.jsonl") as f:
        for line in f:
            outputs.append(line)
else:
    prompts = []
    outputs = []

i = 0
for idx, row in tqdm(df.iterrows(), total=len(df)):
    if i < progress:
        i += 1
        continue
    prompt = make_prompt(row)
    prompts.append(prompt)
    model_input = eval_tokenizer(prompt, return_tensors="pt").to("cuda")
    output = eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=1024)[0], skip_special_tokens=True)
    outputs.append(output)
    i += 1
    if i % 50 == 0:
        # save progress
        with open("progress.txt", "w") as f:
            f.write(str(i))
        # save prompts
        with jsonlines.open("prompts.jsonl", "w") as f:
            f.write_all(prompts)
        # save outputs
        with jsonlines.open("outputs.jsonl", "w") as f:
            f.write_all(outputs)
        



# add to dataframe
df["prompt"] = prompts
df["output"] = outputs

# save to file
df.to_csv("data/test_base_model_output.csv", index=False)
# save to jsonl
df.to_json("data/test_base_model_output.jsonl", orient="records", lines=True)

# command to run script in background
# python test_base_model.py > logs/test_base_model.log 2>&1 &
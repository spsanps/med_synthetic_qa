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

from peft import PeftModel

ft_path = (
    "train_outputs/medqa_mistral_dolphin_2_2_1-medqa_mixed_qa_train_20k/checkpoint-750"
)

ft_model = PeftModel.from_pretrained(model, ft_path)

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


from tqdm import tqdm

prompts = []
# reasons = []
outputs = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = make_prompt(row)
    prompts.append(prompt)
    model_input = eval_tokenizer(prompt, return_tensors="pt").to("cuda")
    output = eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=1024)[0], skip_special_tokens=True)
    outputs.append(output)
    


# add to dataframe
df["prompt"] = prompts 
df["output"] = outputs

# save to file
df.to_csv("data/test_ft_qa_model_output.csv", index=False)
# save to jsonl
df.to_json("data/test_ft_qa_model_output.jsonl", orient="records", lines=True)

# command to run script in background
# python test_ft_model.py > logs/test_ft_qa_model.log 2>&1 &
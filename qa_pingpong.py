# %%
# 3 types of synthetic dataset
# 1. Question and Answer (direct facts from text)
# 2. Insight (synthesize new information with its knowledge and text)
# 3. What is the metadata, context. Is it changing? 

# This notebook will be for 1. Question and Answer (direct facts from text)

# %% [markdown]
# # Dataset

# %%
import os

# Dictionary to hold file names and their contents
files_content_dict = {}

# Directory containing the .txt files
directory = './data/en/'

# Check if directory exists
if not os.path.exists(directory):
    print(f"The directory {directory} does not exist.")
else:
    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Check if it is a file and has a .txt extension
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            # Open and read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read the contents of the file
                content = file.read()
                # Add to dictionary with the file name (without extension) as the key
                files_content_dict[os.path.splitext(filename)[0]] = content

# Printing the content dict just for demonstration purposes
for file_name, file_content in files_content_dict.items():
    print(f"{file_name}: {file_content[:50]}...")  # Print the first 50 characters of each file

# You can now use 'files_content_dict' as needed


# %% [markdown]
# #### Let's break down the dataset into groups of 5 paragraphs each

# %%
def split_into_para_groups(text, n = 3, max_length_group = 128*7):
    """
    Split the text into groups of n paragraphs each.
    If the length of the group exceeds max_length_group into groups of length <= max_length_group using for loop to add segments
    and NOT checking the length of the group at each step.
    because this might add a paragraph that exceeds the max_length_group.
    """
    para = text.split("\n")
    para = [p.strip() for p in para if p.strip() != ""]
    para_max_len_opt = []
    for p in para:
        if len(p) > max_length_group:
            for i in range(0, len(p), max_length_group):
                para_max_len_opt.append(p[i:i+max_length_group])
        else:
            para_max_len_opt.append(p)
    para = para_max_len_opt

    para_groups = []
    for i in range(0, len(para), n):
        para_groups.append("\n".join(para[i:i+n]))

    
    return para_groups
    


# %%

paragraphs = []
for file_name, file_content in files_content_dict.items():
    # join 3 paragraphs together
    paragraphs.extend(split_into_para_groups(file_content, n = 3, max_length_group = 128*5))

# %%
print(paragraphs[0])
len(paragraphs)

# %%
print(len(paragraphs[0]))

# %%
import pandas as pd

df = pd.DataFrame(paragraphs, columns=["paragraph"])
# add a column for length of paragraph

df["length"] = df["paragraph"].apply(lambda x: len(x))

# describe the length of pa
# ragraphs
df["length"].describe()

# %%
# find the longest paragraph


sample = df[df["length"] == df["length"].max()]["paragraph"].values[0]
print(sample)
print(len(sample))

# %%
from vllm import LLM, SamplingParams
import vllm
import torch
from typing import List, Callable, Optional
from vllm.sampling_params import SamplingParams
from vllm.model_executor.input_metadata import InputMetadata


# %%
#base_model_id = "ehartford/dolphin-2.0-mistral-7b"
#base_model_id = "HuggingFaceH4/zephyr-7b-alpha"
#base_model_id = "amazon/MistralLite"
base_model_id = "ehartford/dolphin-2.2.1-mistral-7b"
llm = LLM(model=base_model_id)


# %%
template = """<|im_start|>system
You will be asked questions which you can only answer with the information below:
## Context ##
{}
#############
You will accurately answer these questions which will be directly about the information in the context. You will only be asked one question at a time. <|im_end|>
<|im_start|>user
"""

sampling_params = SamplingParams(temperature=0.3, max_tokens=512)

# %%
def generate_prompt(text):
    return template.format(text)

# %%
prompts = [generate_prompt(p) for p in paragraphs]

# %%
print(prompts[0])

# %%
# Model

# %%
#outputs = llm.generate(prompts, sampling_params)

# %%
#output_text = [output.outputs[0].text for output in outputs]

# %%
#print(output_text[789])

# %%
# add paragraphs and prompts and outputs to a dataframe

"""df["prompt"] = prompts
df["output"] = output_text

df.head()

# save the dataframe to a csv file
df.to_csv("data/qa_dataset1.csv", index=False)
"""

# %%
"""new_prompts = [p + o for p, o in zip(prompts, output_text)]
new_prompts = [p.strip() + "\n<|im_start|>assistant\n" for p in new_prompts]
print(new_prompts[0])"""

# %%
# let's do back and forth conversation
# we will keep track of the output

new_prompts = prompts
df = pd.DataFrame(new_prompts, columns=["prompt"])


for i in range(5):
    outputs = llm.generate(new_prompts, sampling_params)
    output_text = [output.outputs[0].text for output in outputs]
    new_prompts = [p + o for p, o in zip(new_prompts, output_text)]
    new_prompts = [p.strip() + "<|im_end|>\n<|im_start|>assistant\n" for p in new_prompts]
    #prompts = new_prompts
    # add to dataframe
    df["prompt"] = new_prompts
    df["output"] = output_text
    df.to_csv(f"data/qa_dataset{i}.csv", index=False)
    outputs = llm.generate(new_prompts, sampling_params)
    output_text = [output.outputs[0].text for output in outputs]
    new_prompts = [p + o for p, o in zip(new_prompts, output_text)]
    new_prompts = [p.strip() + "<|im_end|>\n<|im_start|>user\n" for p in new_prompts]
    #prompts = new_prompts
    # add to dataframe
    df["prompt"] = new_prompts
    df["output"] = output_text
    df.to_csv(f"data/qa_dataset{i}.csv", index=False)
    print(f"Done with {i} iteration")

# %%
print(new_prompts[2])

# %%




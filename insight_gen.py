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

# create a dict of file_name and para_groups

files_para_groups_dict = {}

for file_name, file_content in files_content_dict.items():
    files_para_groups_dict[file_name] = split_into_para_groups(file_content, n = 3, max_length_group = 128*7)

# %%
print(files_para_groups_dict.keys())            

# %%
files_para_groups_dict['Physiology_Levy'][:2]

# %%
# conext bot

from vllm import LLM, SamplingParams
import vllm
import torch
from typing import List, Callable, Optional
from vllm.sampling_params import SamplingParams
from vllm.model_executor.input_metadata import InputMetadata
#base_model_id = "ehartford/dolphin-2.0-mistral-7b"
#base_model_id = "HuggingFaceH4/zephyr-7b-alpha"
#base_model_id = "amazon/MistralLite"
base_model_id = "ehartford/dolphin-2.2.1-mistral-7b"
llm = LLM(model=base_model_id)


# %%
template = """<|im_start|>system
You will be figuring out what the context for the following section from book: {book} is about. 
You will be provided with previous context - update it as necessary or change it if it's a new section. You are expected to give the book name - and a short description of what chapter you are in (you can figure it out from the previous context) <|im_end|>
<|im_start|>user
## Previous Context
{previous_context} 
## Current Content
{text} <|im_end|>
<|im_start|>assistant
"""

assistant_preprompt = "The contex of the current content is: "

sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

# %%
def generate_prompt(previous_context, book, text):
    return template.format(previous_context = previous_context, book = book, Text = text) + assistant_preprompt


# %%
def get_next_contexts(previous_contexts, books, texts):
    prompts = [generate_prompt(previous_context, book, text) for previous_context, book, text in zip(previous_contexts, books, texts)]
    outputs = llm.generate(prompts, sampling_params)
    contexts = [output.outputs[0].text for output in outputs]
    return contexts


# %%
"""# genrate paragraph by paragraph for each file
# they have different lengths but we still need to genereate parallely

previous_context = ["Current section is the first section. No previous context."]*len(files_para_groups_dict.keys())

books = list(files_para_groups_dict.keys())

active_contexts = {}
active_contexts = {book:"Current section is the first section. No previous context." for book in books}

for i in range(0, 10):
    previous_context = [active_contexts[book] for book in books]
    books = list(active_contexts.keys())
    texts = [files_para_groups_dict[book][i] for book in books]
    contexts = get_next_contexts(previous_context, books, texts)
    previous_context = contexts
    for book, context in zip(books, contexts):
        raise notImplementedError"""

# %%
# Template 2 insight

template = """<|im_start|>system
You will be provided with a some content. You will make deeper insights into this content and connections with knowledge you might already have. The Insight should be understandable without any context (stand alone). You will be response will be a terse but dense paragraph that covers all important insights and context about the content provided. <|im_end|>
<|im_start|>user
{content} <|im_end|>
<|im_start|>assistant
Context, Insights and Connections 
"""

sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

# %%
def generate_prompt(content):
    return template.format(content = content) + "Insight: "



# %%
# infer all the insights

insights = {}

for book, paras in files_para_groups_dict.items():
    prompts = [generate_prompt(para) for para in paras]
    outputs = llm.generate(prompts, sampling_params)
    insights[book] = [output.outputs[0].text for output in outputs]
    # save the insights to a file
    with open(f"./data/insights/{book}.txt", "w") as f:
        f.write("\n".join(insights[book]))

# flatten the insights
flat_insights = []
file_names = []
for book, ins in insights.items():
    flat_insights.extend(ins)
    file_names.extend([book]*len(ins))
# save as csv with df

import pandas as pd

df = pd.DataFrame({"file_name":file_names, "insights":flat_insights})

df.to_csv("./data/insights.csv", index = False)




# %%
df.head()

# %%
insights

# %%


# %%




# Importing the required packages
import pandas as pd
from PIL import Image
import os
from matplotlib import pyplot as plt
import yaml
from datagen import MMQADataset
from torch.utils.data import DataLoader
from google import genai
from questionDecomposition import PromptGeneration
import open_clip
import torch
import re
from ImageSelection import ImageSelection
from answerGeneration import queryGeneration, answerGeneration, captionGeneration
from tqdm import tqdm
import time

# Set up CLIP (open_clip version)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')


# Loading the client to model to generate question decomposition
client = genai.Client(api_key="")
# gemini_model = 'gemini-1.5-pro'
gemini_model = 'gemini-1.5-pro'

# Load the config.yaml file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract paths into variables
MMQA_dev = config['data_paths']['MMQA_dev']
MMQA_train = config['data_paths']['MMQA_train']
MMQA_test = config['data_paths']['MMQA_test']

MMQA_images = config['data_paths']['MMQA_images']
MMQA_tables = config['data_paths']['MMQA_tables']
MMQA_texts = config['data_paths']['MMQA_texts']

image_root_path = config['root_path']
intermediate_results_path = config['intermediate_results_path']
final_results_path = config['final_results_path']


# Loading the dataset and the creating a dataloader
# Defining a Custom collate function since pytorch cannot handle dataframes directly
def custom_collate_fn(batch):
    # batch is a list of items returned by __getitem__
    return batch 

dataset = MMQADataset(jsonl_path=MMQA_dev, jsontab_path=MMQA_tables, jsoni_path=MMQA_images, jsontxt_path=MMQA_texts, image_root=image_root_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Final results for appending
final_df = pd.DataFrame(columns=['Question', 'Answer', 'Generated Answer'])
# Looping through the samples
for i, a_sample in enumerate(tqdm(dataloader)):
    table, image_associated_columns, question, answer, metadata = a_sample[0]

    try:
        # STEP 1: QUESTION DECOMPOSITION
        # Getting the prompt and the label
        prompt = PromptGeneration(table, image_associated_columns, question, metadata)
        # Getting the response
        response = client.models.generate_content(model=gemini_model, contents=prompt)
        output = response.text
        # Getting the composition and the corresponding ImageQ and TableQ (this part needs working)
        composition = output.split('\n')[0]
        # print("Composition: ", composition)
        imageq_match = re.search(r'ImageQ:\s*(.*)', output)
        if imageq_match:
            imageQ = imageq_match.group(1)
            # print("ImageQ:", imageQ)
        
        tableq_match = re.search(r'TableQ:\s*(.*)', output)
        if tableq_match:
            tableQ = tableq_match.group(1)
            # print("TableQ: ", tableQ)

        # STEP 2: RELEVANT IMAGES
        # Getting the list of images
        images = table.dropna(subset=['ImagePath'])['ImagePath'].values.tolist()
        k = 10 # Select top 3 three images
        top_images = ImageSelection(image_root_path, preprocess, tokenizer, model, device, imageQ, images, k)
        # Pruning the table based on this (Have to change based on OS)
        req_images = [i.split('\\')[-1] for i in top_images]
        pruned_table = table[table['ImagePath'].isin(req_images)]
        
        # STEP 3: REPLACE IMAGES WITH CAPTIONS
        pruned_captioned_table = captionGeneration(pruned_table, image_root_path, imageQ, client, gemini_model)

        # STEP 4: GENERATE QUERY WITH FINAL QUESTION
        query = queryGeneration(pruned_captioned_table, question)
        generated_answer = answerGeneration(pruned_captioned_table, question,  client, gemini_model)

        # Saving intermediate results
        # First create a folder
        example_path = os.path.join(intermediate_results_path, f'Example_{i}')
        if not os.path.exists(example_path):
            os.makedirs(example_path)

        # Save decomposition result
        with open(os.path.join(example_path, 'decomposition.txt'), 'w+') as f:
            f.writelines(output)
        # Saving the pruned table
        pruned_table.to_csv(os.path.join(example_path, 'pruned_table.csv'))
        # Saving the pruned decomposed table
        pruned_captioned_table.to_csv(os.path.join(example_path, 'pruned_captioned_table.csv'))

        # Saving the final results
        final_df.loc[i, 'Question'] = question
        final_df.loc[i, 'Answer'] = answer
        final_df.loc[i, 'Generated Answer'] = generated_answer.text
    
    except Exception as e:
        example_path = os.path.join(intermediate_results_path, f'Example_{i}')
        if not os.path.exists(example_path):
            os.makedirs(example_path)
        with open(os.path.join(example_path, 'error.txt'), 'w+') as f:
            f.writelines(str(e))

    final_df.to_csv(os.path.join(final_results_path, 'results_k15.csv'), index=True)
    # time.sleep(20)

    if i> 50:
        break




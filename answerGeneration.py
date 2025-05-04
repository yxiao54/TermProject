import pandas as pd
import base64
import os

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def captionGeneration(pruned_table, root_path, imageq, client, model):
    # Generating the caption
    return_pruned_table = pruned_table.copy()
    for ind in pruned_table.index.values.tolist():
        image_path = os.path.join(root_path, pruned_table.loc[ind, 'ImagePath'])
        response = client.models.generate_content(
        model=model,
        contents=[
            {"role": "user", "parts": [
                {"text": f"Given this question: '{imageq}', give me a single line caption for the image that answers the question"},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": encode_image(image_path)  # you must read and encode the image
                        }}
                    ]}
                ]
            )
        caption = response.text
        return_pruned_table.loc[ind, imageq] = caption

    return return_pruned_table
    

def queryGeneration(pruned_table, question):
    # Convert Table to dict
    table_dict = pruned_table.to_dict()
    prompt = f'Given the following question and table, generate an SQL QUERY \n Question: {question} Table: {table_dict}'

    return prompt

def answerGeneration(pruned_table, question, client, gemini_model):
    # Convert Table to dict
    table_dict = pruned_table.to_dict()
    prompt = f'\n Question:" {question} Table: {table_dict} \n Answer the question in a single line based on the given question and the table'

    response = client.models.generate_content(model=gemini_model, contents=prompt)

    return response

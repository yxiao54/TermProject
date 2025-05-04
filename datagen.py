import os
import json
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def composition(df):
    df.loc[:, 'Composition'] = df.loc[:, 'metadata'].apply(lambda x: x['type'])
    return_df = df[df['Composition'].isin(['Compose(ImageQ,TableQ)'])]
    return_df = return_df.drop(columns=['Composition'])
    return_df = return_df.reset_index()

    return return_df


class MMQADataset(Dataset):
    def __init__(self, jsonl_path, jsontab_path, jsoni_path, jsontxt_path, image_root=None, transform=None):
        """
        Args:
            jsonl_path (str): Path to the .jsonl file containing data.
            image_root (str, optional): Path to the directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dev = pd.read_json(jsonl_path, lines=True)
        # Add a function here to have only images and table questions and answers
        self.dev = composition(self.dev)
        self.tables = pd.read_json(jsontab_path, lines=True)
        self.images = pd.read_json(jsoni_path, lines=True)
        self.texts = pd.read_json(jsontxt_path, lines=True)
        self.image_root = image_root
        self.transform = transform
    


    def generate_table(self, tables_df, table_id):
        # Getting the table columns and rows
        table = tables_df[tables_df['id']==table_id]['table'].iloc[0]
        table_rows = table['table_rows']
        table_name = table['table_name']
        table_columns = table['header']

        # Looping through each column
        table_dict = dict()
        image_associated_columns = list()
        for i, a_col in enumerate(table_columns):
            table_dict[a_col['column_name']] = list()
            # Identifying the image associated column
            if 'image_associated_column' in list(a_col['metadata'].keys()):
                if table_columns[a_col['metadata']['image_associated_column']]:
                    image_associated_columns.append(a_col['column_name'])
            for a_row in table_rows:
                table_dict[a_col['column_name']].append(a_row[i]['text'])

        generated_table = pd.DataFrame(table_dict)

        return generated_table, image_associated_columns


    # Add Images and Text to the table
    def combined_table(self, images_df, texts_df, generated_table, image_associated_columns, metadata):

        # Getting the associated Image IDs and Text IDs
        image_ids = metadata['image_doc_ids']
        text_ids = metadata['text_doc_ids']

        req_text_df = texts_df[texts_df['id'].isin(text_ids)].copy()

        # Editing the images df function
        images_df['title'] = images_df['title'].apply(lambda x: x.split('(')[0])
        req_images_df = images_df[images_df['id'].isin(image_ids)].copy()
        # Creating a temporary merge column
        req_images_df['mergeColumn'] = req_images_df['title'].apply(lambda x: x.replace(' ','').lower())
        # Combining the image associated column
        combined_df = generated_table.copy()
        if len(image_associated_columns) == 1:
            combined_df.loc[:,'mergeColumn'] = combined_df[image_associated_columns[0]].apply(lambda x: x.replace(' ', '').lower())
            combined_df = pd.merge(combined_df, req_images_df, on='mergeColumn', how='left')

        combined_df = combined_df.drop(columns=['mergeColumn', 'url', 'id', 'title'])
        combined_df = combined_df.rename(columns={'path': 'ImagePath'})

        return combined_df, req_text_df

    def __len__(self):
        return len(self.dev)

    def __getitem__(self, ind):
        """
        Args:
            ind (int): Index of the sample.

        Returns:
            final_table: The table with associated image paths embedded
            image_associated_columns: The columns which have associated images
            question: the question
            answer: the corresponding answer
            metadata: accompanying metadata
        """
        a_sample = self.dev.loc[ind]
        qid = a_sample['qid']
        question = a_sample['question']
        answer = a_sample['answers'][0]['answer']
        # Getting the metadata and the supporting context
        metadata = a_sample['metadata']
        supporting_context = a_sample['supporting_context']

        # Getting the corresponding table ID
        table_id = metadata['table_id']
        generated_table, image_associated_columns = self.generate_table(self.tables, table_id)
        # Getting the final table
        final_table, ass_text_table = self.combined_table(self.images, self.texts, generated_table, image_associated_columns, metadata)


        return final_table, image_associated_columns, question, answer, metadata  # or return { "input": input_tensor, "label": label, ... }



if __name__ == '__main__':
    pass
    
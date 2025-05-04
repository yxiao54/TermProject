import os
import pandas as pd
import sqlite3
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import open_clip
import torch
from google import genai


def ImageSelection(IMAGE_BASE_PATH, preprocess, tokenizer, model, device, imageq, images, k):
    # --- Embedding + Ranking ---
    def get_text_embedding(text):
        with torch.no_grad():
            tokenized = tokenizer([text]).to(device)
            return model.encode_text(tokenized) / model.encode_text(tokenized).norm(dim=-1, keepdim=True)

    def get_image_embedding(image_path):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_image(image) / model.encode_image(image).norm(dim=-1, keepdim=True)

    def get_top_k_images(imageq, image_filenames, k=3):
        text_embed = get_text_embedding(imageq)
        img_embeds = []
        valid_paths = []

        for fname in image_filenames:
            full_path = os.path.join(IMAGE_BASE_PATH, os.path.basename(fname))
            if os.path.exists(full_path):
                img_embeds.append(get_image_embedding(full_path).cpu().numpy()[0])
                valid_paths.append(full_path)

        similarities = cosine_similarity(text_embed.cpu().numpy(), img_embeds)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [valid_paths[i] for i in top_k_indices]
    
    return get_top_k_images(imageq, images, k)

if __name__ == '__main__':
    pass
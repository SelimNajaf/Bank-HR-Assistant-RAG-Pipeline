"""
Module to demonstrate basic text embedding generation using Hugging Face's sentence-transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModel

def generate_embedding(text: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> torch.Tensor:
    """
    Generates a sentence embedding for a given text using a pre-trained Hugging Face model.

    Args:
        text (str): The input text to be embedded.
        model_name (str): The Hugging Face model repository name.

    Returns:
        torch.Tensor: The mean-pooled sentence embedding tensor.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input text
    encoded_input = tokenizer(text, return_tensors="pt")

    # Generate model outputs
    with torch.no_grad():
        outputs = model(**encoded_input)

    # Perform mean pooling on the last hidden state to get the sentence embedding
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    
    return sentence_embedding


if __name__ == "__main__":
    sample_text = "Kapital Bankda nahar fasiləsi 13:00-da başlayır."
    embedding = generate_embedding(sample_text)
    
    print(f"Text: '{sample_text}'")
    print(f"Embedding shape: {embedding.shape}")
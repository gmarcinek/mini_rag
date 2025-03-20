import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class PolishLegalEmbedder:
    def __init__(self, use_gpu: bool = False, model_name = "BAAI/bge-m3"):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print(f"Załadowano model {model_name}")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        return sum_embeddings / sum_mask.clamp(min=1e-9)

    def get_embedding(self, text: str) -> np.ndarray:
        # Tokenizacja z automatycznym paddingiem i truncation (do 8192 tokenów)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=8192
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = self.mean_pooling(outputs, inputs['attention_mask'])

        return embedding.cpu().numpy()

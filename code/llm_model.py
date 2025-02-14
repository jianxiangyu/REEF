from torch import Tensor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class SentenceEncoder:
    def __init__(self, llm_name, device, cache_dir="../cache_data/model", multi_gpu=False):
        self.llm_name = llm_name
        self.device = device
        self.multi_gpu = multi_gpu
        self.cache_dir = cache_dir

    def encode(self, input_texts: list[str]) -> Tensor:
        '''
            input_texts: list of strings, length of input_texts is n 
            output: n*d
        '''
        if input_texts is None:
            raise ValueError("The input texts cannot be empty")
        
        if self.llm_name == 'Sentence-Bert':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            all_embeddings = model.encode(input_texts, convert_to_tensor=True)
            all_embeddings = F.normalize(all_embeddings, dim=1).cpu()
        return all_embeddings
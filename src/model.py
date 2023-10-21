from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class SkipgramModel(nn.Module):
    def __init__(self, vocab_size: int, 
                 vocabulary: List[str],
                 word_to_idx: Dict, 
                 embedding_dim: int, 
                 noise_distribution: torch.Tensor) -> None:
        super().__init__()
        
        self.vocabulary = vocabulary
        self.vocab_size = vocab_size
        self.word_to_idx = word_to_idx
        
        self.embedding_dim = embedding_dim
        self.noise_distribution = noise_distribution
        
        # layers
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.output_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # uniform init
        self.word_embedding.weight.data.uniform_(-1, 1)
        self.output_embedding.weight.data.uniform_(-1, 1)
        
    
    def generate_noise(self, batch_size: int, 
                       device: torch.device, 
                       n_samples: int=5) -> torch.Tensor:
        noise_word_idxs = torch.multinomial(self.noise_distribution, 
                                        num_samples=batch_size * n_samples, replacement=True)
        noise_word_idxs = noise_word_idxs.to(device)
        noise = self.output_embedding(noise_word_idxs)
        
        # reshape
        noise = rearrange(noise, "(batch samples) embed -> batch samples embed", 
                          batch=batch_size, samples=n_samples)
        
        return noise
        
    def forward(self, 
                target: torch.Tensor, 
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_vectors = self.word_embedding(target)
        context_vectors = self.output_embedding(context)
        
        return target_vectors, context_vectors
        


class NegativeSamplingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, target: torch.Tensor, context: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # rearrange for matrix multiplication
        target = rearrange(target, "batch embed -> batch embed 1")
        context = rearrange(context, "batch embed -> batch 1 embed")
        
        
        word_embedding_loss = torch.bmm(context, target).sigmoid().log().squeeze()
        noise_loss = torch.bmm(noise.neg(), target).sigmoid().log().squeeze()
        
        
        total_loss = - word_embedding_loss.neg() - noise_loss.sum(dim=-1)
        mean_loss = total_loss.mean(dim=-1)
        
        return mean_loss
        
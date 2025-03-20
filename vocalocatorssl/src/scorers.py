import torch
from torch import nn


class Scorer(nn.Module):
    def __init__(self, d_audio_embed: int, d_location_embed: int):
        """Abstract class for scoring affinity between audio and location embeddings.

        Args:
            d_audio_embed (int): Dimension of the audio embeddings
            d_location_embed (int): Dimension of the location embeddings
        """
        super(Scorer, self).__init__()
        self.d_audio_embed = d_audio_embed
        self.d_location_embed = d_location_embed

    def forward(
        self, audio_embedding: torch.Tensor, location_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Computes the affinity scores between audio and location embeddings.
        Assumes both batch dimensions are the same or broadcastable.

        Args:
            audio_embedding (torch.Tensor): Audio embeddings (*batch, d_audio_embedding)
            location_embedding (torch.Tensor): Location embeddings (*batch, d_location_embedding)

        Returns:
            torch.Tensor: Affinity scores.
        """
        raise NotImplementedError("Forward method must be implemented in subclass.")


class CosineSimilarityScorer(Scorer):
    def __init__(self, d_embedding: int):
        super(CosineSimilarityScorer, self).__init__(d_embedding, d_embedding)

    def forward(
        self, audio_embedding: torch.Tensor, location_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Computes the cosine similarity between audio and location embeddings.

        Args:
            audio_embedding (torch.Tensor): Audio embeddings (*batch, d_embedding)
            location_embedding (torch.Tensor): Location embeddings (*batch, d_embedding)

        Returns:
            torch.Tensor: Cosine similarity scores.  (*batch, )
        """
        audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
        location_embedding = location_embedding / location_embedding.norm(
            dim=-1, keepdim=True
        )
        return (audio_embedding * location_embedding).sum(dim=-1)


class MLPScorer(Scorer):
    def __init__(self, d_audio_embed: int, d_location_embed: int, d_hidden: int):
        """MLP based scorer for audio-location affinity.

        Args:
            d_audio_embed (int): Dimension of the audio embeddings
            d_location_embed (int): Dimension of the location embeddings
            d_hidden (int): Dimension of the hidden layer
        """
        super(MLPScorer, self).__init__(d_audio_embed, d_location_embed)
        self.mlp = nn.Sequential(
            nn.Linear(d_audio_embed + d_location_embed, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, audio_embedding: torch.Tensor, location_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Computes the affinity scores between audio and location embeddings.
        Assumes both embeddings have the same batch dimensions.

        Args:
            audio_embedding (torch.Tensor): Audio embeddings (*batch, d_audio_embedding)
            location_embedding (torch.Tensor): Location embeddings (*batch, d_location_embedding)

        Returns:
            torch.Tensor: Affinity scores. (*batch, )
        """
        x = torch.cat([audio_embedding, location_embedding], dim=-1)
        return self.mlp(x).squeeze(-1)

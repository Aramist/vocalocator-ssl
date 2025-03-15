from typing import Literal

import numpy as np
import torch
from torch import nn

from .profiling import record


class LocationEmbedding(nn.Module):
    def __init__(
        self,
        d_location: int,
        d_embedding: int,
        n_expected_locations: int,
    ):
        """An abstract class for modules that convert one or more spatial coordinates
        into a fixed-size embedding.
        """
        super().__init__()

        self.d_location = d_location
        self.d_embedding = d_embedding
        self.num_locations = n_expected_locations

    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        """Generates a fixed-size embedding from a batch of location coordinates.

        Args:
            locations (torch.Tensor): Locations of shape (*batch, num_locations, num_nodse, num_dims)

        Raises:
            NotImplementedError: This method should be implemented by subclasses

        Returns:
            torch.Tensor: Embeddings of shape (*batch, d_embedding)
        """
        raise NotImplementedError("forward method must be implemented by subclasses")


class FourierEmbedding(LocationEmbedding):
    def __init__(
        self,
        d_location: int,
        d_embedding: int,
        n_expected_locations: int,
        *,
        init_bandwidth: float = 1,
        location_combine_mode: Literal["concat", "add"] = "concat",
        **kwargs,  # Don't crash if we get extra kwargs
    ):
        """Creates a learned positional embedding based on Fourier Features.
        The bandwidth is provided in the same units as the locations.

        Args:
            d_location (int): Dimensionality of the location coordinates
            d_embedding (int): Dimensionality of the embedding
            n_expected_locations (int): Number of locations included in each embedding
            init_bandwidth (float, optional): Bandwidth of the random projection's initial weights. Defaults to 1.

        Raises:
            ValueError: If d_embedding is not even
            ValueError: If location_combine_mode is not 'concat' or 'add'
            ValueError: If init_bandwidth is non-positive
        """
        super(FourierEmbedding, self).__init__(
            d_location, d_embedding, n_expected_locations
        )
        if d_embedding % 2 != 0:
            raise ValueError(f"d_embedding must be even, got {d_embedding}")
        if location_combine_mode not in ["concat", "add"]:
            raise ValueError(
                f"location_combine_mode must be 'concat' or 'add', got {location_combine_mode}"
            )
        if init_bandwidth <= 0:
            raise ValueError(f"init_bandwidth must be positive, got {init_bandwidth}")

        self.bandwidth = init_bandwidth
        self.location_combine_mode = location_combine_mode

        d_input = (
            d_location * n_expected_locations
            if location_combine_mode == "concat"
            else d_location
        )
        self.rand_projection = nn.Parameter(
            torch.zeros(d_input, d_embedding // 2), requires_grad=True
        )

        self.rng = np.random.default_rng()

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.rand_projection, mean=0, std=1 / self.bandwidth)

    @record
    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        """Generates a fixed-size embedding from a batch of location coordinates.

        Args:
            locations (torch.Tensor): (*batch, num_locations, num_nodes, num_dims) tensor of location coordinates

        Raises:
            ValueError: If combine mode is 'concat' and the number of locations is not equal to n_expected_locations

        Returns:
            torch.Tensor: (*batch, d_embedding) tensor of embeddings
        """
        if self.location_combine_mode == "concat":
            # flatten the num_locations, num_nodes, and num_dims into one
            if locations.shape[-2] != self.num_locations:
                raise ValueError(
                    f"Expected {self.num_locations} locations, got {locations.shape[-2]}"
                )
            # Shuffle the locations to avoid overfitting to the order
            perm = self.rng.permutation(self.num_locations)
            locations = locations[:, perm]
            locations = locations.view(*locations.shape[:-3], -1)
            # resulting shape: (*batch, d_location * num_locations)
        else:
            # flatten the num_nodes and num_dims into one
            locations = locations.view(*locations.shape[:-2], -1)
            # resulting shape: (*batch, num_locations, d_location)

        mapping = torch.einsum("...m,md->...d", locations, self.rand_projection)
        # if concat: shape is (*batch, d_embed)
        # if add: shape is (*batch, num_locations, d_embed)
        if self.location_combine_mode == "add":
            # Honestly not sure if it makes more sense for this to come before or after the sincos
            mapping = mapping.mean(dim=-2)

        return torch.cat([torch.cos(mapping), torch.sin(mapping)], dim=-1) / np.sqrt(
            self.d_embedding
        )


class MLPEmbedding(LocationEmbedding):
    def __init__(
        self,
        d_location: int,
        d_embedding: int,
        n_expected_locations: int,
        *,
        n_hidden: int = 1,
        hidden_dim: int = 512,
        location_combine_mode: Literal["concat", "add"] = "concat",
        **kwargs,  # Don't crash if we get extra kwargs
    ):
        """Creates a simple MLP for transforming 3d locations into a fixed-size embedding.

        Args:
            d_location (int): Dimensionality of the location coordinates
            d_embedding (int): Dimensionality of the output embedding
            n_expected_locations (int): Number of locations embedded together
            n_hidden (int, optional): Number of hidden layers in the MLP. Defaults to 1.
            hidden_dim (int, optional): Dimensionality of the hidden layers. Defaults to 512.
            location_combine_mode (Literal[&quot;concat&quot;, &quot;add&quot;], optional): How the locations are combined into the embedding

        Raises:
            ValueError: If d_embedding is not even
            ValueError: If location_combine_mode is not 'concat' or 'add'
        """
        super(MLPEmbedding, self).__init__(
            d_location, d_embedding, n_expected_locations
        )

        # Check inputs
        if d_embedding % 2 != 0:
            raise ValueError(f"d_embedding must be even, got {d_embedding}")
        if location_combine_mode not in ["concat", "add"]:
            raise ValueError(
                f"location_combine_mode must be 'concat' or 'add', got {location_combine_mode}"
            )

        self.hidden_dim = hidden_dim
        self.n_layers = n_hidden
        self.location_combine_mode = location_combine_mode

        # Calculate input dimensionality
        input_dim = (
            d_location * n_expected_locations
            if location_combine_mode == "concat"
            else d_location
        )

        # Create MLP
        channel_sizes = [input_dim] + [hidden_dim] * n_hidden + [d_embedding]
        self.dense = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU())
                for d_in, d_out in zip(channel_sizes[:-1], channel_sizes[1:])
            ],
        )

    @record
    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        """Creates a fixed-size embedding from a batch of location coordinates.

        Args:
            locations (torch.Tensor): (*batch, num_locations, num_nodes, num_dims) tensor of location coordinates

        Returns:
            torch.Tensor: (*batch, d_embedding) tensor of embeddings
        """
        if self.location_combine_mode == "concat":
            # flatten the num_locations, num_nodes, and num_dims into one
            locations = locations.view(*locations.shape[:-3], -1)
        else:
            # flatten the num_nodes and num_dims into one
            locations = locations.view(*locations.shape[:-2], -1)

        embed = self.dense(locations)
        # if concat: shape is (*batch, d_embed)
        # if add: shape is (*batch, num_locations, d_embed)

        if self.location_combine_mode == "add":
            embed = embed.sum(dim=-2)
        return embed

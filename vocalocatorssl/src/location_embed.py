from typing import Literal

import numpy as np
import torch
from torch import nn


class LocationEmbedding(nn.Module):
    def __init__(
        self,
        d_location: int,
        d_embedding: int,
        multinode_strategy: Literal["absolute", "relative"] = "absolute",
    ):
        """An abstract class for modules that convert one or more spatial coordinates
        into a fixed-size embedding.

        Args:
            d_location (int): Dimensionality of the location coordinates
            d_embedding (int): Dimensionality of the output embedding
            multinode_strategy (Literal["absolute", "relative"], optional): How to handle multiple nodes. Defaults to "absolute".
                - "absolute": Each node is treated independently.
                - "relative": All nodes except the first are made relative to the first node.
        """
        super().__init__()

        self.d_location = d_location
        self.d_embedding = d_embedding
        self.multinode_strategy = multinode_strategy
        if multinode_strategy not in ["absolute", "relative"]:
            raise ValueError(
                f"multinode_strategy must be 'absolute' or 'relative', got {multinode_strategy}"
            )

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
        *,
        init_bandwidth: float = 1,
        multinode_strategy: Literal["absolute", "relative"] = "absolute",
        **kwargs,  # Don't crash if we get extra kwargs
    ):
        """Creates a learned positional embedding based on Fourier Features.
        The bandwidth is provided in the same units as the locations.

        Args:
            d_location (int): Dimensionality of the location coordinates
            d_embedding (int): Dimensionality of the embedding
            init_bandwidth (float, optional): Bandwidth of the random projection's initial weights. Defaults to 1.
            multinode_strategy (Literal["absolute", "relative"], optional): How to handle multiple nodes. Defaults to "absolute".
                - "absolute": Each node is treated independently.
                - "relative": All nodes except the first are made relative to the first node.

        Raises:
            ValueError: If d_embedding is not even
            ValueError: If location_combine_mode is not 'concat' or 'add'
            ValueError: If init_bandwidth is non-positive
        """
        super(FourierEmbedding, self).__init__(
            d_location, d_embedding, multinode_strategy
        )
        if d_embedding % 2 != 0:
            raise ValueError(f"d_embedding must be even, got {d_embedding}")
        if init_bandwidth <= 0:
            raise ValueError(f"init_bandwidth must be positive, got {init_bandwidth}")

        self.bandwidth = init_bandwidth

        d_input = d_location
        self.rand_projection = nn.Parameter(
            torch.zeros(d_input, d_embedding // 2), requires_grad=True
        )

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.rand_projection, mean=0, std=1 / self.bandwidth)

    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        """Generates a fixed-size embedding from a batch of location coordinates.

        Args:
            locations (torch.Tensor): (*batch, num_locations, num_nodes, num_dims) tensor of location coordinates

        Raises:
            ValueError: If combine mode is 'concat' and the number of locations is not equal to n_expected_locations

        Returns:
            torch.Tensor: (*batch, d_embedding) tensor of embeddings
        """
        if self.multinode_strategy == "relative":
            # Make all nodes except the first node relative to the first
            locations[..., 1:, :] -= locations[..., :1, :]
            locations[..., 1:, :] /= torch.linalg.norm(
                locations[..., 1:, :], dim=-1, keepdim=True
            )

        # flatten the num_nodes and num_dims into one
        locations = locations.view(*locations.shape[:-2], -1)
        # resulting shape: (*batch, num_animals, num_locations, d_location)

        mapping = torch.einsum("...m,md->...d", locations, self.rand_projection)
        # shape is (*batch, num_locations, d_embed)

        return torch.cat([torch.cos(mapping), torch.sin(mapping)], dim=-1) / np.sqrt(
            self.d_embedding
        )


class MLPEmbedding(LocationEmbedding):
    def __init__(
        self,
        d_location: int,
        d_embedding: int,
        *,
        n_hidden: int = 1,
        hidden_dim: int = 512,
        multinode_strategy: Literal["absolute", "relative"] = "absolute",
        **kwargs,  # Don't crash if we get extra kwargs
    ):
        """Creates a simple MLP for transforming 3d locations into a fixed-size embedding.

        Args:
            d_location (int): Dimensionality of the location coordinates
            d_embedding (int): Dimensionality of the output embedding
            n_expected_locations (int): Number of locations embedded together
            n_hidden (int, optional): Number of hidden layers in the MLP. Defaults to 1.
            hidden_dim (int, optional): Dimensionality of the hidden layers. Defaults to 512.
            multinode_strategy (Literal["absolute", "relative"], optional): How to handle multiple nodes. Defaults to "absolute".
                - "absolute": Each node is treated independently.
                - "relative": All nodes except the first are made relative to the first node.

        Raises:
            ValueError: If d_embedding is not even
            ValueError: If location_combine_mode is not 'concat' or 'add'
        """
        super(MLPEmbedding, self).__init__(d_location, d_embedding, multinode_strategy)

        # Check inputs
        if d_embedding % 2 != 0:
            raise ValueError(f"d_embedding must be even, got {d_embedding}")

        self.hidden_dim = hidden_dim
        self.n_layers = n_hidden

        # Calculate input dimensionality
        input_dim = d_location

        # Create MLP
        channel_sizes = [input_dim] + [hidden_dim] * n_hidden + [d_embedding]
        self.dense = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU())
                for d_in, d_out in zip(channel_sizes[:-1], channel_sizes[1:])
            ],
        )

    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        """Creates a fixed-size embedding from a batch of location coordinates.

        Args:
            locations (torch.Tensor): (*batch, num_locations, num_nodes, num_dims) tensor of location coordinates

        Returns:
            torch.Tensor: (*batch, d_embedding) tensor of embeddings
        """
        if self.multinode_strategy == "relative":
            # Make all nodes except the first node relative to the first
            locations[..., 1:, :] -= locations[..., :1, :]
            locations[..., 1:, :] /= torch.linalg.norm(
                locations[..., 1:, :], dim=-1, keepdim=True
            )

        # flatten the num_nodes and num_dims into one
        locations = locations.view(*locations.shape[:-2], -1)

        embed = self.dense(locations)
        # shape is (*batch, num_locations, d_embed)

        return embed


class MixedEmbedding(LocationEmbedding):
    def __init__(
        self,
        d_location: int,
        d_embedding: int,
        *,
        xy_poly_degree: int = 5,
        z_poly_degree: int = 4,
        angle_fourier_degree: int = 32,
        d_hidden: int = 128,
        num_layers: int = 2,
        init_bandwidth: float = 0.3,
    ):
        """Creates an embedding that incorporates prior beliefs about each input dimension
        Args:
            d_location (int): Dimensionality of the location coordinates
            d_embedding (int): Dimensionality of the output embedding
        """
        # The multinode strategy is not actually used in this class
        super(MixedEmbedding, self).__init__(
            d_location, d_embedding, multinode_strategy="absolute"
        )
        self.k = xy_poly_degree
        self.l = z_poly_degree
        self.m = angle_fourier_degree
        self.d_embedding = d_embedding
        self.d_location = d_location  # Ignore this. Assuming we always have 3D x 2 locations (head + nose; x,y,z)

        layers = [
            nn.Linear((self.k + 1) ** 2 + (self.l + 1) + (2 * self.m), d_hidden),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(d_hidden, d_embedding))
        self.mlp = nn.Sequential(*layers)

        self.fourier_linear = nn.Linear(
            2,
            self.m,
        )

        nn.init.normal_(self.fourier_linear.weight, mean=0, std=1 / init_bandwidth)

    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        """Generates a fixed-size embedding from a batch of location coordinates.

        Args:
            locations (torch.Tensor): (*batch, num_nodes=2, num_dims=3) tensor of location coordinates

        Returns:
            torch.Tensor: (*batch, d_embedding) tensor of embeddings
        """
        if locations.shape[-2] != 2 or locations.shape[-1] != 3:
            raise ValueError(
                f"locations must have shape (*batch, num_nodes=2, num_dims=3), got {locations.shape}"
            )

        xy = locations[..., 0, :2].cpu().numpy()
        z = locations[..., 0, 2].cpu().numpy()
        head_direction = locations[..., 0, :] - locations[..., 1, :]
        theta = torch.atan2(head_direction[..., 1], head_direction[..., 0])
        phi = torch.atan2(
            head_direction[..., 2], torch.linalg.norm(head_direction[..., :2], dim=-1)
        )
        thetaphi = torch.stack([theta, phi], dim=-1)  # (*batch, 2)

        # Create the polynomial features
        xy_feats = np.polynomial.legendre.legvander2d(
            xy[..., 0], xy[..., 1], (self.k, self.k)
        )  # (*batch, (k+1)**2)
        xy_feats = torch.from_numpy(xy_feats).to(torch.float32).to(locations.device)
        z_feats = np.polynomial.legendre.legvander(z, self.l)  # (*batch, (l+1))
        z_feats = torch.from_numpy(z_feats).to(torch.float32).to(locations.device)

        angle_mapping = self.fourier_linear(thetaphi)
        angle_feats = torch.cat(
            [torch.cos(angle_mapping), torch.sin(angle_mapping)], dim=-1
        ) / np.sqrt(2 * self.m)

        features = torch.cat([xy_feats, z_feats, angle_feats], dim=-1)
        # shape is (*batch, (k+1)**2 + (l+1) + 2*m)

        return self.mlp(features)


if __name__ == "__main__":
    # Test the MixedEmbedding class
    locations = torch.randn(10, 2, 3)  # 10 samples, 2 nodes, 3 dimensions
    embedding = MixedEmbedding(d_location=3, d_embedding=128)
    output = embedding(locations)
    print(output.shape)  # Should be (10, 128)

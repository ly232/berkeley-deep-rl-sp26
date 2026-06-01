"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn

from itertools import pairwise
from jaxtyping import Float


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""

    def scaffold_mlp(
        self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int
    ) -> nn.Sequential:
        """Helper function to build an MLP with ReLU activations."""
        layers = []
        for l1, l2 in pairwise((input_dim,) + hidden_dims):
            layers.extend([nn.Linear(l1, l2), nn.ReLU()])
        layers.append(
            nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, output_dim)
        )
        return nn.Sequential(*layers)


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        input_dim = state_dim
        output_dim = action_dim * chunk_size
        self.model = self.scaffold_mlp(input_dim, hidden_dims, output_dim)

    def forward(
        self, state: Float[torch.Tensor, "batch, state_dim"]
    ) -> Float[torch.Tensor, "batch, chunk_size, action_dim"]:
        out = self.model(state)
        return out.reshape(
            state.shape[0],  # batch size
            self.chunk_size,
            self.action_dim,
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        preds = self(state)  # (batch, chunk, action)
        loss = nn.functional.mse_loss(preds, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            action_chunk = self(state)
        return action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        input_dim = state_dim + action_dim * chunk_size + 1  # +1 for time step
        output_dim = action_dim * chunk_size
        self.model = self.scaffold_mlp(input_dim, hidden_dims, output_dim)

    def forward(
        self,
        state: Float[torch.Tensor, "batch, state_dim"],
        interpolation: Float[torch.Tensor, "batch, chunk_size, action_dim"],
        tau: Float[torch.Tensor, "batch 1 1"],
    ) -> Float[torch.Tensor, "batch, chunk_size, action_dim"]:
        """Inference the *velocity*, not the action chunk."""
        batch_size = state.shape[0]
        tau_flat = tau.reshape(batch_size, 1)
        interpolation_flat = interpolation.reshape(
            interpolation.shape[0], self.action_dim * self.chunk_size
        )
        input = torch.cat([state, interpolation_flat, tau_flat], dim=1)
        out = self.model(input)
        return out.reshape(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        noise = torch.randn_like(
            action_chunk, device=state.device, dtype=action_chunk.dtype
        )
        tau = torch.rand(
            (batch_size, 1, 1), device=action_chunk.device, dtype=action_chunk.dtype
        )
        interpolation = tau * action_chunk + (1 - tau) * noise
        velocity_preds = self(state, interpolation, tau)
        velocity_targets = action_chunk - noise
        loss = nn.functional.mse_loss(velocity_preds, velocity_targets)
        return loss

    def inference(
        self,
        state: Float[torch.Tensor, "batch, state_dim"],
        num_steps: int,
    ) -> Float[torch.Tensor, "batch, chunk_size, action_dim"]:
        """Run the flow model in inference mode by integrating the velocity."""
        batch_size = state.shape[0]
        action_chunk = torch.randn(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        for step in range(num_steps):
            tau = torch.full(
                (batch_size, 1, 1),
                step / num_steps,
                device=state.device,
                dtype=state.dtype,
            )
            dt = 1.0 / num_steps
            velocity = self(state, action_chunk, tau)
            action_chunk = action_chunk + velocity * dt
        return action_chunk

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.inference(state, num_steps=num_steps)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

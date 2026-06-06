import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO(done): implement get_action
        # Convert observation to tensor. Note we unsqueeze(0) for batch dim.
        obs = torch.tensor(obs, dtype=torch.float32, device=ptu.device).unsqueeze(0)
        distribution = self.forward(obs)
        action = distribution.sample().squeeze(0)

        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO(done): define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)  # shape: (batch_size, num_actions)
            # Convert to probabilities.
            return distributions.Categorical(logits=logits)
        else:
            # TODO(done): define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            # Convert to probabilities.
            return distributions.Normal(mean, std)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO(done): compute the policy gradient actor loss
        #
        # ATTN: this is different than supervised learning, where we predict
        # outputs then compute loss against golden. Here for policy gradients,
        # and for RL in general, we do NOT have goldens, and instead we compute
        # loss as negative of log prob weighted by reward (remember, we want to
        # gradient ASCENT over log prob weighted by reward, so negate for loss)
        distribution = self.forward(obs)
        if self.discrete:
            log_probs = distribution.log_prob(actions)
        else:
            log_probs = distribution.log_prob(actions).sum(dim=-1)
        loss = -torch.mean(log_probs * advantages)

        # TODO(done): perform an optimizer step
        # for on-policy, samples are thrown away so no point keeping old grads
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }

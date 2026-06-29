from types import SimpleNamespace
from unittest.mock import patch

import torch

from hw4.rl.base import AlgoConfig
from hw4.rl.grpo import GRPO
from hw4.rollout.rollout_buffer import RolloutBatch


class TinyPolicy(torch.nn.Module):
    """Minimal trainable module for testing GRPO.update wiring."""

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.logp_offset = torch.nn.Parameter(torch.tensor(init))


def make_rollout(
    *,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor | None = None,
    ref_logprobs: torch.Tensor | None = None,
) -> RolloutBatch:
    n, logprob_len = completion_mask.shape
    input_len = logprob_len + 1
    if old_logprobs is None:
        old_logprobs = torch.zeros((n, logprob_len))
    if ref_logprobs is None:
        ref_logprobs = torch.zeros((n, logprob_len))
    return RolloutBatch(
        input_ids=torch.zeros((n, input_len), dtype=torch.long),
        attention_mask=torch.ones((n, input_len), dtype=torch.long),
        completion_mask=completion_mask.float(),
        old_logprobs=old_logprobs.float(),
        ref_logprobs=ref_logprobs.float(),
        rewards=torch.zeros(n),
        advantages=advantages.float(),
    )


def test_grpo_update_steps_parameter_for_unclipped_positive_advantage():
    model = TinyPolicy(init=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    rollout = make_rollout(
        completion_mask=torch.tensor([[1.0, 1.0]]),
        advantages=torch.tensor([1.0]),
        old_logprobs=torch.tensor([[-0.5, -1.5]]),
        ref_logprobs=torch.tensor([[-0.5, -1.5]]),
    )
    algo = GRPO(
        AlgoConfig(
            ppo_epochs=1,
            minibatch_size=1,
            clip_eps=0.2,
            kl_coef=0.0,
            max_grad_norm=10.0,
            adv_clip=10.0,
        )
    )
    fixed_logp = torch.tensor([[-0.5, -1.5]])

    def fake_compute_per_token_logprobs(model, input_ids, attention_mask):
        del input_ids, attention_mask
        return fixed_logp + model.logp_offset

    before = model.logp_offset.detach().clone()
    with patch(
        "hw4.rl.grpo.compute_per_token_logprobs",
        side_effect=fake_compute_per_token_logprobs,
    ) as mock_logp:
        stats = algo.update(model=model, optimizer=optimizer, rollout=rollout)

    mock_logp.assert_called_once()
    assert model.logp_offset.item() > before.item()
    assert stats["train/count_optimizer_steps_per_training_iteration"] == 1.0
    assert (
        stats[
            "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches"
        ]
        == 0.0
    )


def test_grpo_update_logs_clip_fraction_when_ratio_outside_clip_range():
    model = TinyPolicy(init=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    rollout = make_rollout(
        completion_mask=torch.tensor([[1.0, 1.0]]),
        advantages=torch.tensor([1.0]),
        old_logprobs=torch.tensor([[-2.0, -1.0]]),
        ref_logprobs=torch.tensor([[0.0, 0.0]]),
    )
    algo = GRPO(
        AlgoConfig(
            ppo_epochs=1,
            minibatch_size=1,
            clip_eps=0.2,
            kl_coef=0.0,
            max_grad_norm=10.0,
            adv_clip=10.0,
        )
    )
    fixed_logp = torch.tensor([[0.0, -1.0]])

    def fake_compute_per_token_logprobs(model, input_ids, attention_mask):
        del input_ids, attention_mask
        return fixed_logp + 0.0 * model.logp_offset

    with patch(
        "hw4.rl.grpo.compute_per_token_logprobs",
        side_effect=fake_compute_per_token_logprobs,
    ):
        stats = algo.update(model=model, optimizer=optimizer, rollout=rollout)

    torch.testing.assert_close(
        torch.tensor(
            stats[
                "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches"
            ]
        ),
        torch.tensor(0.5),
    )


def test_grpo_update_skips_empty_completion_mask():
    model = TinyPolicy(init=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    rollout = make_rollout(
        completion_mask=torch.tensor([[0.0, 0.0]]),
        advantages=torch.tensor([1.0]),
    )
    algo = GRPO(AlgoConfig(ppo_epochs=1, minibatch_size=1))

    with patch("hw4.rl.grpo.compute_per_token_logprobs") as mock_logp:
        stats = algo.update(model=model, optimizer=optimizer, rollout=rollout)

    mock_logp.assert_not_called()
    assert stats["train/count_optimizer_steps_per_training_iteration"] == 0.0
    assert (
        stats["train/count_minibatches_skipped_because_completion_mask_had_no_tokens"]
        == 1.0
    )
